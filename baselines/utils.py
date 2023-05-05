import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
import fasttext
import fsspec

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


def load_uids_with_basic_filter_helper(path):

    df = pd.read_parquet(
        path, columns=["uid", "text", "original_width", "original_height"]
    )

    df["fasttext_lang_pred"] = df.text.apply(lambda x: get_language(x))
    df["caption_num_words"] = df.text.apply(lambda x: get_words_count(x))
    df["caption_num_chars"] = df.text.apply(lambda x: get_chars_count(x))
    uid_int = df.uid.apply(int, base=16)
    df["uid_upper_uint64"] = (uid_int // 2**64).astype("uint64")
    df["uid_lower_uint64"] = (uid_int % 2**64).astype("uint64")

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


def save_uids(uids, path):
    uids.sort()
    np.save(path, uids)
