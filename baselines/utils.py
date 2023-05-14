import hashlib
import os
import random
import urllib
import warnings
from multiprocessing import Pool
from typing import Any, List

import numpy as np
import torch
from tqdm import tqdm


def random_seed(seed: int = 0) -> None:
    """set seed

    Args:
        seed (int, optional): seed value. Defaults to 0.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def download(name: str, root: str = None) -> str:
    """dowload assets necessary for running baselines (e.g., model weights)

    Args:
        name (str): name key of the asset to download
        root (str, optional): cache location to download the asset. Defaults to None.

    Raises:
        ValueError: unsupported name
        RuntimeError: file exists but it is not a normal file
        RuntimeError: file exists but has the incorrect sha256 checksum

    Returns:
        str: _description_
    """
    # modified from oai _download clip function

    if root is None:
        root = os.path.expanduser("~/.cache/datacomp")
    else:
        root = os.path.expanduser(root)

    # paths for checkpoints we may need to download for baselines along with their sha256 hashes
    cloud_checkpoints = {
        "imagenet21k_wordnet_ids": {
            "url": "https://github.com/sagadre/datacomp_baseline_assets/releases/download/v0.1.0-alpha/imagenet21k_wordnet_ids.txt",
            "sha256": "66362bdedf36d933382edca5493fc562dcc17128ce36403c9e730a75f48cb2f2",
        },
        "in1k_clip_vit_l14_0": {
            "url": "https://github.com/sagadre/datacomp_baseline_assets/releases/download/v0.1.0-alpha/in1k_clip_vit_l14_0.pt",
            "sha256": "304990fd492f40ba90072b80af78d6e8edcab9d042476ef513e03441cf53743b",
        },
        "in1k_clip_vit_l14_1": {
            "url": "https://github.com/sagadre/datacomp_baseline_assets/releases/download/v0.1.0-alpha/in1k_clip_vit_l14_1.pt",
            "sha256": "72f008e6aa3bfb54174fa322963215337f843fc90fd642a9dccc815dd9e7ee76",
        },
        "in1k_clip_vit_l14_2": {
            "url": "https://github.com/sagadre/datacomp_baseline_assets/releases/download/v0.1.0-alpha/in1k_clip_vit_l14_2.pt",
            "sha256": "83c3cc6503b8a50a22cac7023ca6cd61b130ab9725db2a7519f1bec5cabf5493",
        },
        "in1k_clip_vit_l14_3": {
            "url": "https://github.com/sagadre/datacomp_baseline_assets/releases/download/v0.1.0-alpha/in1k_clip_vit_l14_3.pt",
            "sha256": "83faa289f847799ddcd438e5b74df1cb5973c345674896d10ba31ea48bc63804",
        },
        "in1k_clip_vit_l14_4": {
            "url": "https://github.com/sagadre/datacomp_baseline_assets/releases/download/v0.1.0-alpha/in1k_clip_vit_l14_4.pt",
            "sha256": "067a45f4a140a93371876510a2fd9198a89ac1d194ae4063a51b45a6da69cadf",
        },
        "large_centroids": {
            "url": "https://github.com/sagadre/datacomp_baseline_assets/releases/download/v0.1.0-alpha/large_centroids_l14.pt",
            "sha256": "04eeab1069d3540c246cf7ce69323147351a474017fc472e4c50a018ca32240b",
        },
        "medium_centroids": {
            "url": "https://github.com/sagadre/datacomp_baseline_assets/releases/download/v0.1.0-alpha/medium_centroids_l14.pt",
            "sha256": "028b1721b1d0f139c565b6b0ac99f8a1756f4bae89c36b0ec6d1c6ea9b6f112d",
        },
        "small_centroids": {
            "url": "https://github.com/sagadre/datacomp_baseline_assets/releases/download/v0.1.0-alpha/small_centroids_l14.pt",
            "sha256": "23c66a05e49ad77283c1e2b33355c7eb088ac332a944c97ff85d5dfd48a5b251",
        },
        "xlarge_centroids": {
            "url": "https://github.com/sagadre/datacomp_baseline_assets/releases/download/v0.1.0-alpha/xlarge_centroids_l14.pt",
            "sha256": "3f62e5f8ae3a715ce84e422846fcfce1536d184455ea234790c4a6465c4c6726",
        },
        "fasttext": {
            "url": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
            "sha256": "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e",
        },
    }

    if name not in cloud_checkpoints:
        raise ValueError(
            f"unsupported cloud checkpoint: {name}. currently we only support: {list(cloud_checkpoints.keys())}"
        )

    os.makedirs(root, exist_ok=True)

    expected_sha256 = cloud_checkpoints[name]["sha256"]
    download_target = None
    if name == "fasttext":
        download_target = os.path.join(root, "lid.176.bin")
    else:
        download_target = os.path.join(root, f"{name}.pt")
    url = cloud_checkpoints[name]["url"]

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            == expected_sha256
        ):
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        print(f"downloading {url} to {download_target}")
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if (
        hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target


def worker_threadpool(
    worker_fn: Any, concat_fn: Any, paths: List[str], n_workers: int
) -> np.ndarray:
    """get filtered uids

    Args:
        worker_fn (Any): function to map over the pool
        concat_fn (Any): function to use to collate the results
        paths (List[str]): metadata paths to process
        n_workers (int): number of cpu workers

    Returns:
        np.ndarray: filtered uids
    """
    print("creating thread pool for processing")
    with Pool(n_workers) as pool:
        uids = []
        for u in tqdm(
            pool.imap_unordered(worker_fn, paths),
            total=len(paths),
        ):
            uids.append(u)

    return concat_fn(uids)
