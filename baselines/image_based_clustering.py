"""
This is a command line script for clustering image embeddings for the DataComp pool.
The output of the script is a numpy file containing the computed cluster centers.
Please see image_based_clustering.md for additional information, and note that we also provide precomputed numpy files with the cluster centers used in the DataComp baselines.
"""

import argparse
import multiprocessing as mp
from functools import partial
from multiprocessing import Pool
from typing import Any, List, Tuple

import faiss
import fasttext
import fsspec
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from baselines.apply_filter import caption_filter
from baselines.utils import download, random_seed

torch.backends.cudnn.benchmark = True


def train_kmeans(
    embeddings: np.ndarray, num_clusters: int, num_gpus: int, seed: int = 0
) -> torch.Tensor:
    """train kmeans on embeddings

    Args:
        embeddings (np.ndarray): embeddings to cluster
        num_clusters (int): number of clusters
        num_gpus (int): number of gpus to use
        seed (int, optional): random seed. Defaults to 0.
    """
    d = embeddings.shape[1]
    cluster = faiss.Clustering(d, num_clusters)
    cluster.verbose = True
    cluster.niter = 20
    cluster.seed = seed

    # otherwise the kmeans implementation sub-samples the training set
    cluster.max_points_per_centroid = 10000000

    res = [faiss.StandardGpuResources() for i in range(num_gpus)]

    flat_config = []
    for i in range(num_gpus):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if num_gpus == 1:
        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    else:
        indexes = [
            faiss.GpuIndexFlatL2(res[i], d, flat_config[i]) for i in range(num_gpus)
        ]
        index = faiss.IndexReplicas()
        for sub_index in indexes:
            index.addIndex(sub_index)

    # perform the training
    cluster.train(embeddings, index)
    centroids = faiss.vector_float_to_array(cluster.centroids)

    return centroids.reshape(num_clusters, d)


def load_embedding_helper(
    fs_root: Tuple[Any, str],
    key: str = "l14_img",
    caption_filtering: bool = False,
    sample_ratio: float = -1.0,
) -> np.ndarray:
    """worker function to load embeddings

    Args:
        fs_root (Tuple[Any, str]): (filesystem, path_root)
        key (str, optional): key to load from npz. Defaults to "l14_img".
        caption_filtering (bool, optional): whether to enable caption filter. Defaults to False.
        sample_ratio (float, optional): ratio of samples to use. Defaults to -1.0.
    """

    fs, path_root = fs_root
    embed = np.load(fs.open(f"{path_root}.npz"))[key]
    if caption_filtering:
        lang_detect_model = fasttext.load_model(
            download("fasttext", "~/.cache/fasttext")
        )
        df = pd.read_parquet(
            f"{path_root}.parquet", columns=["uid", "text"], filesystem=fs
        )
        mask = caption_filter(df, lang_detect_model)
        embed = embed[mask]
    if sample_ratio > 0:
        n = len(embed)
        idx = np.random.choice(range(n), size=int(n * sample_ratio))
        embed = embed[idx]
    return embed


def load_embedding(
    paths: List[Tuple[Any, str]],
    n_workers: int = 10,
    key: str = "l14_img",
    caption_filtering: bool = False,
    sample_ratio: float = -1.0,
) -> np.ndarray:
    """worker function to load embeddings

    Args:
        paths (List[Tuple[Any, str]]): list of (filesystem, path_root)
        n_workers (int, optional): number of workers. Defaults to 10.
        key (str, optional): key to load from npz. Defaults to "l14_img".
        caption_filtering (bool, optional): whether to enable caption filter. Defaults to False.
        sample_ratio (float, optional): ratio of samples to use. Defaults to -1.0.
    """
    mp.set_start_method("spawn", force=True)
    print("start loading embedding")
    worker = partial(
        load_embedding_helper,
        key=key,
        caption_filtering=caption_filtering,
        sample_ratio=sample_ratio,
    )

    with Pool(n_workers) as pool:
        embeds = [
            res
            for res in tqdm(
                pool.imap(worker, paths), total=len(paths)
            )  # imap so that it can be reproduced
            if len(res) > 0
        ]
    return np.vstack(embeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_dir",
        type=str,
        help="directory (local or cloud) containing parquet, npz metadata",
    )
    parser.add_argument("--save_path", type=str, help="local path to output centroids")
    parser.add_argument(
        "--num_clusters", default=100000, type=int, help="number of clusters"
    )
    parser.add_argument(
        "--embedding_key",
        default="l14_img",
        type=str,
        choices=["l14_img", "b32_img"],
        help="precomputed embeddings used for clustering",
    )
    parser.add_argument(
        "--sample_ratio",
        default=-1.0,
        type=float,
        help="ratio of samples to use (we need to sample because of memory constraint)",
    )
    parser.add_argument(
        "--num_gpus", default=8, type=int, help="number of gpus used for clustering"
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="number of workers, generally set to number of cpu cores",
    )
    parser.add_argument(
        "--disable_caption_filtering",
        default=False,
        action="store_true",
        help="whether to disable text-based basic filtering",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
    )
    args = parser.parse_args()

    random_seed(args.seed)

    num_clusters = args.num_clusters
    num_gpus = args.num_gpus
    sample_ratio = args.sample_ratio
    caption_filtering = not args.disable_caption_filtering

    fs, url = fsspec.core.url_to_fs(args.metadata_dir)
    paths = [(fs, str(x.split(".parquet")[0])) for x in fs.ls(url) if ".parquet" in x]

    print(f"caption filtering: {caption_filtering} | sample_ratio={sample_ratio}")
    embeddings = load_embedding(
        paths,
        key=args.embedding_key,
        n_workers=args.num_workers,
        caption_filtering=caption_filtering,
        sample_ratio=sample_ratio,
    )
    print(f"done: {len(embeddings)}")

    print(f"start clustering: num_clusters = {num_clusters}, num_gpus = {num_gpus}")
    embeddings = embeddings.astype(np.float32)
    centroids = train_kmeans(
        embeddings, num_clusters, num_gpus=num_gpus, seed=args.seed
    )
    torch.save(centroids, args.save_path, pickle_protocol=4)
