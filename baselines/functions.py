import os
from functools import partial
from multiprocessing import Pool

import fasttext
import fsspec
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import wordnet
from tqdm import tqdm
import torch

nltk.download("wordnet")

fasttext.FastText.eprint = lambda x: None
lang_detect_model = fasttext.load_model(
    os.path.expanduser("~/.cache/fasttext/lid.176.bin")
)


def get_language(text):
    text = text.replace("\n", " ")
    language = lang_detect_model.predict(text)[0][0].split("__label__")[1]
    return language


def get_words_count(text):
    word_list = text.split()
    number_of_words = len(word_list)
    return number_of_words


def get_chars_count(text):
    return len(text)


def text_basic_filter(df):
    df['caption_num_words'] = df.text.apply(lambda x: len(fasttext.tokenize(x)))
    df['caption_num_chars'] = df.text.apply(len)
    lang_preds, _ = lang_detect_model.predict([x.replace('\n', ' ') for x in df.text.values], k=1)
    df['fasttext_lang_pred'] = [x[0].replace('__label__', '') for x in lang_preds]
    mask = ((df['fasttext_lang_pred'] == 'en') & (df['caption_num_words'] > 1) & (df['caption_num_chars'] > 5))
    return mask.to_numpy()

@torch.no_grad()
def score(x, y):
    bs_score = torch.einsum("ik, jk -> ij", x, y)
    index = torch.argmax(bs_score, dim=1)
    index_pt = index.long()
    return index_pt


class runner():
    def __init__(self, feat, center, rank, size=8):
        self.num_local = feat.shape[0] // size + int(rank < feat.shape[0] % size)
        self.start = feat.shape[0] // size * rank + min(rank, feat.shape[0] % size)
        current_feat = feat[self.start: self.start + self.num_local]
        self.feat = torch.from_numpy(current_feat).to(torch.device(rank % size))
        self.center = torch.from_numpy(center).to(torch.device(rank % size))
        self.is_end = False
        self.index = 0

        self.pt_label = torch.zeros(self.feat.size(0), dtype=torch.long, device=self.feat.device)

    def __call__(self, batch_size):
        if self.index + batch_size < self.num_local:
            end = self.index + batch_size
        else:
            end = self.num_local
            self.is_end = True

        x = self.feat[self.index: end]
        y = self.center

        index_pt = score(x, y)
        self.pt_label[self.index: end] = index_pt
        self.index += batch_size


@torch.no_grad()
def get_cluster_labels_gpu(centroids, x, batch_size, ngpu):
    size = ngpu

    list_runner = []

    for rank in range(size):
        list_runner.append(runner(x, centroids, rank, ngpu))

    end_list = [0, ] * size
    while sum(end_list) < size:
        for idx, runner_instance in enumerate(list_runner):
            runner_instance: runner
            if not end_list[idx]:
                runner_instance(batch_size)
            if runner_instance.is_end:
                end_list[idx] = 1

    np_label = np.concatenate([x.pt_label.cpu().numpy() for x in list_runner])
    return np_label




def get_cluster_labels(centroids, x, batch_size=1024):
    labels = []
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        scores = np.einsum("ik, jk -> ij", x_batch, centroids)
        labels_batch = np.argmax(scores, axis=1)
        labels.append(labels_batch)
    labels = np.concatenate(labels)
    return labels


def clustering_filter_helper(path_root, centroids, target_clusters, batch_size=1024):
    df = pd.read_parquet(
        f'{path_root}.parquet', columns=["uid", "text"]
    )
    candidate_embedding = np.load(f'{path_root}.npz')['l14_img']

    # modified version of basic filtering first
    mask = text_basic_filter(df)

    uids = df.uid[mask]

    candidate_labels = get_cluster_labels(centroids, candidate_embedding[mask], batch_size=batch_size)
    cluster_to_uids = {}
    for uid, label in zip(uids, candidate_labels):
        cluster_to_uids.setdefault(label, []).append(uid)

    uids_to_keep = []
    for cluster in target_clusters:
        if cluster in cluster_to_uids:
            uids_to_keep.extend(cluster_to_uids[cluster])
    return np.array([(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids_to_keep], np.dtype("u8,u8"))


def load_uids_with_basic_filter_helper(path):
    df = pd.read_parquet(
        path, columns=["uid", "text", "original_width", "original_height"]
    )

    df["fasttext_lang_pred"] = df.text.apply(lambda x: get_language(x))
    df["caption_num_words"] = df.text.apply(lambda x: get_words_count(x))
    df["caption_num_chars"] = df.text.apply(lambda x: get_chars_count(x))
    uid_int = df.uid.apply(int, base=16)
    df["uid_upper_uint64"] = (uid_int // 2 ** 64).astype("uint64")
    df["uid_lower_uint64"] = (uid_int % 2 ** 64).astype("uint64")

    df = df.drop("uid", axis=1)
    df = df.drop("text", axis=1)

    inds_array = np.array(list(zip(df.uid_upper_uint64, df.uid_lower_uint64)), "u8,u8")

    english_mask = df.fasttext_lang_pred == "en"
    caption_mask = (df.caption_num_words > 2) & (df.caption_num_chars > 5)
    min_image_dim = np.minimum(df.original_width, df.original_height)
    max_image_dim = np.maximum(df.original_width, df.original_height)
    aspect_ratio = max_image_dim / min_image_dim
    image_mask = (min_image_dim >= 200) & (aspect_ratio <= 3.0)

    return inds_array[english_mask & caption_mask & image_mask]


def does_contain_text_entity(text, entity_set):
    word_list = text.split()
    all_relevant_synsets_in_sentence = set()
    for word in word_list:
        synsets = wordnet.synsets(word)
        if len(synsets) == 0:
            continue

        synset = synsets[0]
        synset_key = synset.offset()
        if synset_key in entity_set:
            all_relevant_synsets_in_sentence.add(synset_key)

    return len(all_relevant_synsets_in_sentence) != 0


def load_uids_with_text_entity_helper(path, entity_set):
    df = pd.read_parquet(path, columns=["uid", "text"])

    df["fasttext_lang_pred"] = df.text.apply(lambda x: get_language(x))
    df["contains_in21k_synset"] = df.text.apply(
        lambda x: does_contain_text_entity(x, entity_set)
    )

    uid_int = df.uid.apply(int, base=16)
    df["uid_upper_uint64"] = (uid_int // 2**64).astype("uint64")
    df["uid_lower_uint64"] = (uid_int % 2**64).astype("uint64")

    df = df.drop("uid", axis=1)
    df = df.drop("text", axis=1)

    inds_array = np.array(list(zip(df.uid_upper_uint64, df.uid_lower_uint64)), "u8,u8")

    english_mask = df.fasttext_lang_pred == "en"
    in21k_mask = df.contains_in21k_synset == True

    return inds_array[english_mask & in21k_mask]


def load_uids_with_threshold_helper(path, key, threshold):
    df = pd.read_parquet(path, columns=["uid", key])
    return np.array(
        [
            (int(uid[:16], 16), int(uid[16:32], 16))
            for uid in df[df[key] >= threshold]["uid"].values
        ],
        np.dtype("u8,u8"),
    )


def load_metadata(path, n_workers, columns=None):
    fs, url = fsspec.core.url_to_fs(path)
    paths = [str(x) for x in fs.ls(url) if ".parquet" in x]

    worker = partial(pd.read_parquet, columns=columns)
    with Pool(n_workers) as pool:
        df = pd.concat(list(pool.imap_unordered(worker, paths)))
    return df


def get_threshold(paths, key, p=0.3, n_workers=50):
    df = load_metadata(paths, n_workers=n_workers, columns=[key])
    n = int(len(df) * p)
    threshold = -np.sort(-df[key].values)[n]

    return threshold


def load_uids_with_threshold(path, key, threshold, n_workers):
    fs, url = fsspec.core.url_to_fs(path)
    paths = [str(x) for x in fs.ls(url) if ".parquet" in x]

    worker = partial(load_uids_with_threshold_helper, key=key, threshold=threshold)
    with Pool(n_workers) as pool:
        uids = []
        for u in tqdm(pool.imap_unordered(worker, paths), total=len(paths)):
            uids.append(u)

    return np.concatenate(uids)


def load_uids_with_basic_filter(path, n_workers):
    fs, url = fsspec.core.url_to_fs(path)
    paths = [str(x) for x in fs.ls(url) if ".parquet" in x]

    with Pool(n_workers) as pool:
        uids = []
        for u in tqdm(
                pool.imap_unordered(load_uids_with_basic_filter_helper, paths),
                total=len(paths),
        ):
            uids.append(u)

    return np.concatenate(uids)


def load_uids_with_text_entity(path, n_workers):
    entity_ids = open("./baselines/assets/imagenet22k_synsets.txt", "r").readlines()
    entity_ids = [x.strip() for x in entity_ids]
    entity_ids = [int(x[1:]) for x in entity_ids]

    fs, url = fsspec.core.url_to_fs(path)
    paths = [str(x) for x in fs.ls(url) if ".parquet" in x]

    worker = partial(load_uids_with_text_entity_helper, entity_set=entity_ids)

    with Pool(n_workers) as pool:
        uids = []
        for u in tqdm(
            pool.imap_unordered(worker, paths),
            total=len(paths),
        ):
            uids.append(u)

    return np.concatenate(uids)

def load_uids_with_image_filtering(path, scale, n_workers):

    prefix = None
    if scale == 'small':
        prefix = '10m'
    elif scale == 'medium':
        prefix = '100m'
    elif scale == 'large':
        prefix = '1b'
    elif scale == 'xlarge':
        prefix = '10b'
    else:
        raise ValueError(f'unsupported scale: {scale}')

    target_embedding = torch.load('baselines/assets/in1k_vit_l14.pt')
    centroids = torch.load(f'baselines/assets/{prefix}_centroids.pt')

    target_cluster_labels = get_cluster_labels(centroids, target_embedding, batch_size=batch_size)
    target_clusters = np.unique(target_cluster_labels)

    fs, url = fsspec.core.url_to_fs(path)
    paths = [str(x.split('.parquet')[0]) for x in fs.ls(url) if ".parquet" in x]

    worker = partial(
        clustering_filter_helper,
        centroids=centroids,
        target_clusters=target_clusters
    )

    with Pool(n_workers) as pool:
        uids = []
        for u in tqdm(
            pool.imap_unordered(worker, paths),
            total=len(paths),
        ):
            uids.append(u)

    return np.concatenate(uids)

def save_uids(uids, path):
    uids.sort()
    np.save(path, uids)
