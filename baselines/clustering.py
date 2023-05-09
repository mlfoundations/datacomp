import argparse
from functools import partial
from multiprocessing import Pool

import faiss
import fsspec
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

from functions import text_basic_filter


def train_kmeans(x, k, ngpu):
    "Runs kmeans on one or several GPUs"
    d = x.shape[1]
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.niter = 20

    # otherwise the kmeans implementation sub-samples the training set
    clus.max_points_per_centroid = 10000000

    res = [faiss.StandardGpuResources() for i in range(ngpu)]

    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if ngpu == 1:
        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i])
                   for i in range(ngpu)]
        index = faiss.IndexReplicas()
        for sub_index in indexes:
            index.addIndex(sub_index)

    # perform the training
    clus.train(x, index)
    centroids = faiss.vector_float_to_array(clus.centroids)

    return centroids.reshape(k, d)


def read_embeds_with_sample(path, key='l14_img', basic_filtering=False, sample_ratio=-1):
    parquet_path, npz_path = path
    embed = np.load(npz_path)[key]
    if basic_filtering:
        df = pd.read_parquet(str(parquet_path), columns=['uid', 'text'])
        mask = text_basic_filter(df)
        embed = embed[mask]
    if sample_ratio > 0:
        n = len(embed)
        idx = np.random.choice(range(n), size=int(n * sample_ratio))
        embed = embed[idx]
    return embed


def load_embeds_with_sample(paths, n_workers=10, key='l14_img', basic_filtering=False, sample_ratio=-1):
    print('start loading embedding')
    worker = partial(read_embeds_with_sample, key=key, basic_filtering=basic_filtering, sample_ratio=sample_ratio)

    with Pool(n_workers) as pool:
        embeds = [res for res in tqdm(pool.imap_unordered(worker, paths), total=len(paths)) if len(res) > 0]
    return np.vstack(embeds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_dir', type=str, help="directory (local or cloud) containing parquet, npz metadata")
    parser.add_argument('--save_path', type=str, help="local path to output centroids")
    parser.add_argument('--num_clusters', default=100000, type=int, help="number of clusters")
    parser.add_argument('--sample_ratio', default=-1.0, type=float, help="ratio of samples to use (we need to sample because of memory constraint)")
    parser.add_argument('--num_gpus', default=8, type=int, help="number of gpus used for clustering")
    parser.add_argument('--num_workers', default=10, type=int, help="number of workers, generally set to number of cpu cores")
    parser.add_argument('--basic_filtering', default=False, action="store_true", help="whether to use text-based basic filtering")
    args = parser.parse_args()

    k = args.num_clusters
    ngpu = args.num_gpus
    save_dir = args.save_dir
    sample_ratio = args.sample_ratio

    fs, url = fsspec.core.url_to_fs(args.metadata_dir)
    paths = [str(x.split('.parquet')[0]) for x in fs.ls(url) if ".parquet" in x]
    paths = [(f'{x}.parquet', f'{x}.npz') for x in paths]

    print(f'basic filtering: {args.basic_filtering} | sample_ratio={sample_ratio}')
    X = load_embeds_with_sample(paths, n_workers=args.num_workers, basic_filtering=args.basic_filtering, sample_ratio=sample_ratio)
    print(f'done: {len(X)}')

    print(f'start clustering: k = {k}, num_gpus = {ngpu}')
    X = X.astype(np.float32)
    centroids = train_kmeans(X, k, ngpu=ngpu)
    torch.save(centroids, args.save_path, pickle_protocol=4)
