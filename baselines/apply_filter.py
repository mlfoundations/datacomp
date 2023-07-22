import multiprocessing as mp
import os
import time
from functools import partial
from multiprocessing import Pool
from queue import Empty
from typing import Any, List, Set, Tuple, Union

import fasttext
import fsspec
import gcld3
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import wordnet
from tqdm import tqdm

from baselines.utils import download, worker_threadpool

fasttext.FastText.eprint = lambda x: None


def get_fasttext_language(text: str, lang_detect_model: Any) -> str:
    """helper to detect language of a piece of text (fasttext)

    Args:
        text (str): text whose language we want to determing
        lang_detect_model (Any): fasttext model to detect langauge

    Returns:
        str: ISO language code
    """
    text = text.replace("\n", " ")
    language = lang_detect_model.predict(text)[0][0].split("__label__")[1]

    return language


def get_gcld3_language(text: str, gcld3_model) -> str:
    """helper to detect language of a piece of text (gcld3).
    Note: this is only used for our LAION-2B filtering reproduction

    Args:
        text (str): text whose language we want to determing
        lang_detect_model (Any): fasttext model to detect langauge

    Returns:
        str: ISO language code
    """
    text = text.replace("\n", " ")
    language = gcld3_model.FindLanguage(text=text).language

    return language


def caption_filter(df: pd.DataFrame, lang_detect_model: Any) -> np.ndarray:
    """apply a low-level text filter for the image based baseline

    Args:
        df (pd.DataFrame): parquet metadata
        lang_detect_model (Any): fasttext model

    Returns:
        np.ndarray: boolean numpy array containing selected entries
    """
    caption_num_words = df.text.apply(lambda x: len(fasttext.tokenize(x)))
    caption_num_chars = df.text.apply(len)

    lang_preds, _ = lang_detect_model.predict(
        [x.replace("\n", " ") for x in df.text.values], k=1
    )
    fasttext_en = [x[0].replace("__label__", "") == "en" for x in lang_preds]

    mask = fasttext_en & (caption_num_words > 1) & (caption_num_chars > 5)

    return mask.to_numpy()


@torch.no_grad()
def get_centroid_ids_gpu(
    features: torch.Tensor, centroids: torch.Tensor, batch_size: int, device: int
) -> torch.Tensor:
    """assign features to closest centroid

    Args:
        features (torch.Tensor): features to assign to centroids
        centroids (torch.Tensor): reference centroids
        batch_size (int): gpu batch size
        device (int): gpu number

    Returns:
        torch.Tensor: assignment of features to labels
    """
    device_string = f"cuda:{device}"
    centroids_gpu = centroids.to(device_string)
    labels = torch.zeros(features.shape[0], dtype=torch.long)

    for i in range(0, features.shape[0], batch_size):
        similarity = torch.einsum(
            "ik, jk -> ij",
            features[i : i + batch_size, :].float().to(device_string),
            centroids_gpu,
        )
        matches = torch.argmax(similarity, dim=1).cpu()
        labels[i : i + batch_size] = matches.long()

    return labels


def image_filter_helper(
    pool_centroids: torch.Tensor,
    target_centroid_ids: torch.Tensor,
    batch_size: int,
    device_index: int,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    arch: Union[str, None] = None,
    threshold: Union[float, None] = None,
) -> None:
    """worker function to image_based filtering, pulling off a queue of tasks

    Args:
        pool_centroids (torch.Tensor): centroids derived from k-means on a pool (e.g., the small pool)
        target_centroid_ids (torch.Tensor): target centroid indices of interest, only want samples nearest to these centroids
        batch_size (int): gpu batch size for assigning samples loaded from the in_queue to pool centroids
        device_index (int): device on which to run the gpu processing
        in_queue (mp.Queue): task queue with fsspec, metadata path pairs
        out_queue (mp.Queue): output queue to send filtred uids
        arch: (Union[str, None]): If specified, we want to apply a threshold to arch=B/32 or L/14 clip scores. Defaults to None.
        threshold: (Union[float, None]): threshold to apply over arch clip scores. Defaults to None.
    """
    while True:
        fs_root = None
        try:
            fs_root = in_queue.get(timeout=1)
        except Empty:
            # case where the queue is depleated, worker should return
            break

        fs, path_root = fs_root
        lang_detect_model = fasttext.load_model(
            download("fasttext", "~/.cache/fasttext")
        )

        df = None
        df_index = None

        if arch is not None:
            key = "clip_l14_similarity_score"
            if arch == "b32":
                key = "clip_b32_similarity_score"

            df = pd.read_parquet(
                f"{path_root}.parquet", columns=["uid", "text", key], filesystem=fs
            )
            df_index = df[key] >= threshold
            df = df[df_index]
        else:
            df = pd.read_parquet(
                f"{path_root}.parquet", columns=["uid", "text"], filesystem=fs
            )

        candidate_embedding = None
        with fs.open(f"{path_root}.npz") as f:
            candidate_embedding = torch.from_numpy(np.load(f)["l14_img"])

            if df_index is not None:
                candidate_embedding = candidate_embedding[df_index]

        # simple caption filter first
        mask = caption_filter(df, lang_detect_model)

        uids = df.uid[mask]

        candidate_centroid_ids = get_centroid_ids_gpu(
            candidate_embedding[mask],
            pool_centroids,
            batch_size,
            device_index,
        )

        centroid_id_to_uids = {}
        for uid, label in zip(uids, candidate_centroid_ids):
            centroid_id_to_uids.setdefault(label.item(), []).append(uid)

        uids_to_keep = []
        for i in target_centroid_ids:
            if i.item() in centroid_id_to_uids:
                uids_to_keep.extend(centroid_id_to_uids[i.item()])

        out_queue.put(
            np.array(
                [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids_to_keep],
                np.dtype("u8,u8"),
            )
        )


def load_uids_with_basic_filter_helper(fs_url: Tuple[Any, str]) -> np.ndarray:
    """helper to run basic filter on a single parquet

    Args:
        fs_url (Tuple[Any, str]): pair of fsspec file system and parquet url

    Returns:
        np.ndarray: array of uids
    """
    fs, url = fs_url
    df = pd.read_parquet(
        url, columns=["uid", "text", "original_width", "original_height"], filesystem=fs
    )

    lang_detect_model = fasttext.load_model(download("fasttext", "~/.cache/fasttext"))

    fasttext_lang_pred = df.text.apply(
        lambda x: get_fasttext_language(x, lang_detect_model)
    )
    caption_num_words = df.text.apply(lambda x: len(x.split()))
    caption_num_chars = df.text.apply(lambda x: len(x))
    uid_int = df.uid.apply(int, base=16)
    uid_upper_uint64 = (uid_int // 2**64).astype("uint64")
    uid_lower_uint64 = (uid_int % 2**64).astype("uint64")

    inds_array = np.array(list(zip(uid_upper_uint64, uid_lower_uint64)), "u8,u8")

    english_mask = fasttext_lang_pred == "en"
    caption_mask = (caption_num_words > 2) & (caption_num_chars > 5)
    min_image_dim = np.minimum(df.original_width, df.original_height)
    max_image_dim = np.maximum(df.original_width, df.original_height)
    aspect_ratio = max_image_dim / min_image_dim
    image_mask = (min_image_dim >= 200) & (aspect_ratio <= 3.0)

    return inds_array[english_mask & caption_mask & image_mask]


def does_contain_text_entity(text: str, entity_set: Set) -> bool:
    """helper to check if words in text are contained in an entity set

    Args:
        text (str): caption from an image-text pair
        entity_set (Set): set of synset keys we are cross referencing against

    Returns:
        bool: True if any word of text is in the entity set else False
    """
    word_list = text.split()

    for word in word_list:
        synsets = wordnet.synsets(word)
        if len(synsets) == 0:
            continue

        # retrieve the most likely lemma representing the synset
        synset = synsets[0]
        synset_key = synset.offset()
        if synset_key in entity_set:
            return True

    return False


def load_uids_with_text_entity_helper(
    fs_url: Tuple[Any, str], entity_set: Set
) -> np.ndarray:
    """helper for text based filter on a single parquet

    Args:
        fs_url (str): pair of fsspec file system and parquet url
        entity_set (Set): set of synset keys we are referencing against

    Returns:
        np.ndarray: array of uids
    """
    fs, url = fs_url
    lang_detect_model = fasttext.load_model(download("fasttext", "~/.cache/fasttext"))

    df = pd.read_parquet(url, columns=["uid", "text"], filesystem=fs)
    fasttext_lang_pred = df.text.apply(
        lambda x: get_fasttext_language(x, lang_detect_model)
    )
    contains_in21k_synset = df.text.apply(
        lambda x: does_contain_text_entity(x, entity_set)
    )

    uid_int = df.uid.apply(int, base=16)
    uid_upper_uint64 = (uid_int // 2**64).astype("uint64")
    uid_lower_uint64 = (uid_int % 2**64).astype("uint64")

    inds_array = np.array(list(zip(uid_upper_uint64, uid_lower_uint64)), "u8,u8")

    english_mask = fasttext_lang_pred == "en"
    in21k_mask = contains_in21k_synset == True

    return inds_array[english_mask & in21k_mask]


def load_uids_with_clip_score_helper(
    fs_url: Tuple[Any, str], key: str, threshold: float, gcld3_en_filter: bool
) -> np.ndarray:
    """helper to load parquet metadata with a threshold applied to a column

    Args:
        fs_url (Tuple[Any, str]): pair of fsspec file system and parquet url
        key (str): column we are interested in the parquet column store
        threshold (float): threshold value to apply on the key column
        gcld3_en_filter (bool): if ture, apply gcld3 english filtering (used for laion2b filter)

    Returns:
        np.ndarray: array of uids
    """
    fs, url = fs_url
    df = None

    if gcld3_en_filter:
        df = pd.read_parquet(url, columns=["uid", "text", key], filesystem=fs)

        lang_detect_model = gcld3.NNetLanguageIdentifier(
            min_num_bytes=0, max_num_bytes=1000
        )
        gcld3_lang_pred = df.text.apply(
            lambda x: get_gcld3_language(x, lang_detect_model)
        )
        df = df[gcld3_lang_pred == "en"]
    else:
        df = pd.read_parquet(url, columns=["uid", key], filesystem=fs)

    return np.array(
        [
            (int(uid[:16], 16), int(uid[16:32], 16))
            for uid in df[df[key] >= threshold]["uid"].values
        ],
        np.dtype("u8,u8"),
    )


def load_uids_helper(fs_url: Tuple[Any, str]) -> np.ndarray:
    """helper to read a parquet and load the uids

    Args:
        fs_url (Tuple[Any, str]): pair of fsspec file system and parquet url

    Returns:
        np.ndarray: array of uids
    """
    fs, url = fs_url
    df = pd.read_parquet(url, columns=["uid"], filesystem=fs)

    return np.array(
        [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in df["uid"].values],
        np.dtype("u8,u8"),
    )


def load_metadata(
    metadata_dir_path: str, num_workers: int, columns: List[str] = None
) -> pd.DataFrame:
    """load metadata for many parquets

    Args:
        metadata_dir_path (str): directory where metadata is stored
        num_workers (int): number of cpu workers, each of which processes a parquet
        columns (List[str], optional): list of columns to retain from the parquet. Defaults to None.

    Returns:
        pd.DataFrame: loaded parquet columns
    """
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    parquet_paths = [str(x) for x in fs.ls(url) if ".parquet" in x]
    worker = partial(pd.read_parquet, columns=columns, filesystem=fs)

    return worker_threadpool(worker, pd.concat, parquet_paths, num_workers)


def get_threshold(
    metadata_dir_path: str, key: str, fraction: float, num_workers: int
) -> float:
    """compute a threshold given a collection of metadata, a key, and a target fraction of the pool to keep

    Args:
        metadata_dir_path (str): directory where metadata is stored
        key (str): column we are interested in the parquet column store
        fraction (float): top k fraction, represented as a decimal.
        num_workers (int): number of cpu workers, each of which processes a parquet.

    Returns:
        float: threshold value
    """
    print("loading all metadata for threshold computation")
    df = load_metadata(metadata_dir_path, num_workers=num_workers, columns=[key])
    n = int(len(df) * fraction)
    threshold = -np.sort(-df[key].values)[n]

    return threshold


def load_uids_with_clip_score(
    metadata_dir_path: str,
    arch: str,
    threshold: float,
    fraction: float,
    num_workers: int,
    gcld3_en_filter: bool = False,
) -> np.ndarray:
    """load in metadata with a threshold applied

    Args:
        metadata_dir_path (str): directory where metadata is stored
        key (str): column we are interested in the parquet column store
        threshold (float): threshold value to apply on the key column
        fraction (float): fraction value to apply on the key column
        num_workers (int): number of cpu workers, each of which processes a parquet
        gcld3_en_filter (bool): if True, apply gcld3 english filtering (used for laion2b filter)
                                Default False.

    Returns:
        np.ndarray: array of uids
    """

    # NOTE: assumes errors already checked as in baselines.py
    key = "clip_l14_similarity_score"
    if arch == "b32":
        key = "clip_b32_similarity_score"

    if threshold is None:
        # convert a fraction into a threshold
        threshold = get_threshold(metadata_dir_path, key, fraction, num_workers)

    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    parquet_paths = [(fs, str(x)) for x in fs.ls(url) if ".parquet" in x]

    worker = partial(
        load_uids_with_clip_score_helper,
        key=key,
        threshold=threshold,
        gcld3_en_filter=gcld3_en_filter,
    )

    return worker_threadpool(worker, np.concatenate, parquet_paths, num_workers)


def load_uids(metadata_dir_path: str, num_workers: int) -> np.ndarray:
    """load all uids in a metadata containing directory

    Args:
        metadata_dir_path (str): directory where metadata is stored
        num_workers (int): number of cpu workers, each of which processes a parquet

    Returns:
        np.ndarray: array of uids
    """
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    parquet_paths = [(fs, str(x)) for x in fs.ls(url) if ".parquet" in x]

    return worker_threadpool(
        load_uids_helper, np.concatenate, parquet_paths, num_workers
    )


def load_uids_with_basic_filter(metadata_dir_path: str, num_workers: int) -> np.ndarray:
    """basic filter from the datacomp paper

    Args:
        metadata_dir_path (str): directory where metadata is stored
        num_workers (int): number of cpu workers, each of which processes a parquet

    Returns:
        np.ndarray: array of uids
    """
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    parquet_paths = [(fs, str(x)) for x in fs.ls(url) if ".parquet" in x]

    # download fasttext so that all workers dont't try to download at once
    download("fasttext", "~/.cache/fasttext")

    return worker_threadpool(
        load_uids_with_basic_filter_helper, np.concatenate, parquet_paths, num_workers
    )


def load_uids_with_text_entity(metadata_dir_path: str, num_workers: int) -> np.ndarray:
    """text based filter from the datacomp paper

    Args:
        metadata_dir_path (str): directory where metadata is stored
        num_workers (int): number of cpu workers, each of which processes a parquet

    Returns:
        np.ndarray: array of uids
    """
    entity_ids = open(download("imagenet21k_wordnet_ids"), "r").readlines()
    entity_ids = [x.strip() for x in entity_ids]
    entity_ids = [int(x[1:]) for x in entity_ids]

    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    parquet_paths = [(fs, str(x)) for x in fs.ls(url) if ".parquet" in x]

    # download fasttext so that all workers dont't try to download at once
    download("fasttext", "~/.cache/fasttext")

    worker = partial(load_uids_with_text_entity_helper, entity_set=entity_ids)

    return worker_threadpool(worker, np.concatenate, parquet_paths, num_workers)


def load_uids_with_image_filter(
    metadata_dir_path: str,
    image_based_scale: str,
    num_gpus: int,
    batch_size: int,
    arch: Union[str, None] = None,
    threshold: Union[float, None] = None,
    fraction: Union[float, None] = None,
    num_workers: Union[int, None] = None,
) -> np.ndarray:
    """image based filter from the datacomp paper

    Args:
        metadata_dir_path (str): directory where metadata is stored
        image_based_scale (str): datacomp scale, used to load cached centroids for the pool
        num_gpus (int): number of gpu workers, each of which processes parquet, npy pairs
        batch_size (int, optional): gpu batch size for feature clustering. Defaults to 1024.
        arch (Union[str, None], optional): kind of features for clip filtering. Defaults to None.
        threshold (Union[float, None], optional): threshold to apply to clip features. Defaults to None.
        fraction (Union[float, None], optional): top k fraction to apply to clip features. Defaults to None.
        num_workers (Union[int, None], optional): number of cpu works used to load metadata to compute threshold. Defaults to None.

    Raises:
        RuntimeError: raises in case of a queue mishap, should not happen

    Returns:
        np.ndarray: array of uids
    """

    # load ImageNet-1k OpenAI CLIP ViT-L/14 embeddings
    print("loading ImageNet-1k embeddings")
    target_embedding = torch.cat(
        [
            torch.load(download(f"in1k_clip_vit_l14_{i}"))["image_features"]
            for i in tqdm(range(5))
        ],
        dim=0,
    )

    # use pre-cached pool centroids based on the scale
    print("loading pool centroids")
    pool_centroids = torch.from_numpy(
        torch.load(download(f"{image_based_scale}_centroids"))
    )

    target_centroid_ids = get_centroid_ids_gpu(
        target_embedding, pool_centroids, batch_size, 0
    )
    target_centroid_ids = torch.unique(target_centroid_ids)

    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    root_paths = [
        (fs, str(x.split(".parquet")[0])) for x in fs.ls(url) if ".parquet" in x
    ]

    # initializing task queues
    receive_queue = mp.Queue()
    send_queue = mp.Queue()

    for job in root_paths:
        send_queue.put(job)

    if fraction is not None:
        key = "clip_l14_similarity_score"
        if arch == "b32":
            key = "clip_b32_similarity_score"

        threshold = get_threshold(metadata_dir_path, key, fraction, num_workers)

    # download fasttext so that all workers dont't try to download at once
    download("fasttext", "~/.cache/fasttext")

    processes = []
    print("starting gpu workers")
    for worker_index in tqdm(range(num_gpus)):
        p = mp.Process(
            target=image_filter_helper,
            kwargs=dict(
                pool_centroids=pool_centroids,
                target_centroid_ids=target_centroid_ids,
                batch_size=batch_size,
                device_index=worker_index,
                in_queue=send_queue,
                out_queue=receive_queue,
                arch=arch,
                threshold=threshold,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.05)

    print("processing metadata with gpu workers")
    pbar = tqdm(total=len(root_paths))

    # running in main thread
    all_uids = []
    while True:
        # keep checking for jobs finishing and update uids
        try:
            uids = receive_queue.get(timeout=10)
            all_uids.append(uids)
            pbar.update(1)
        except TimeoutError:
            if all(not p.is_alive() for p in processes):
                try:
                    uids = receive_queue.get(timeout=1)
                    all_uids.append(uids)
                    pbar.update(1)
                except TimeoutError:
                    raise RuntimeError("All processes dead but nothing in queue!")
        except Empty:
            pass

        if all(not p.is_alive() for p in processes):
            # case where all workers have exited
            try:
                uids = receive_queue.get(timeout=1)
                all_uids.append(uids)
                pbar.update(1)
            except Empty:
                print("Result queue is empty and all workers have exited")
                break

    pbar.close()
    for p in processes:
        p.join(timeout=20)

    return np.concatenate(all_uids)


def apply_filter(args: Any) -> None:
    """function to route the args to the proper baseline function

    Args:
        args (Any): commandline args

    Raises:
        ValueError: unsupported name
    """
    mp.set_start_method("spawn", force=True)

    uids = None
    print(f"running: {args.name}")

    if args.name == "no_filter":
        uids = load_uids(
            args.metadata_dir,
            args.num_workers,
        )
    elif args.name == "basic_filter":
        uids = load_uids_with_basic_filter(
            args.metadata_dir,
            args.num_workers,
        )
    elif args.name == "text_based":
        nltk.download("wordnet")
        uids = load_uids_with_text_entity(
            args.metadata_dir,
            args.num_workers,
        )
    elif args.name == "image_based":
        uids = load_uids_with_image_filter(
            args.metadata_dir,
            args.image_based_scale,
            args.num_gpus,
            args.batch_size,
        )
    elif args.name.startswith("clip_score"):
        print(f"threshold {args.threshold} and fraction {args.fraction}")
        uids = load_uids_with_clip_score(
            args.metadata_dir,
            args.arch,
            args.threshold,
            args.fraction,
            args.num_workers,
        )
    elif args.name == "image_based_intersect_clip_score":
        print(f"threshold {args.threshold} and fraction {args.fraction}")
        uids = load_uids_with_image_filter(
            args.metadata_dir,
            args.image_based_scale,
            args.num_gpus,
            args.batch_size,
            arch=args.arch,
            threshold=args.threshold,
            fraction=args.fraction,
            num_workers=args.num_workers,
        )
    elif args.name == "laion2b":
        # special case for laion2b filtering
        uids = load_uids_with_clip_score(
            args.metadata_dir,
            "b32",
            0.28,
            None,
            args.num_workers,
            gcld3_en_filter=True,
        )
    else:
        raise ValueError(f"Unknown args.name argument: {args.name}")

    print(f"sorting {len(uids)} uids")
    uids.sort()

    print(f"saving {args.save_path} with {len(uids)} entries")
    np.save(args.save_path, uids)
