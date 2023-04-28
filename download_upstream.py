import argparse
import os
import re
import shutil

from pathlib import Path
from cloudpathlib import CloudPath

import img2dataset

from huggingface_hub import snapshot_download

from scale_configs import available_scales


def path_or_cloudpath(s):
    if re.match(r"^\w+://", s):
        return CloudPath(s)
    return Path(s)


HF_REPO = 'mlfoundations/datacomp_pools'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--scale', type=str, required=False, choices=available_scales(simple_names=True)[1:], default='small', help='Competition scale.')
    parser.add_argument('--data_dir', type=path_or_cloudpath, required=True, help='Path to directory where the data (webdataset shards) will be stored.')
    parser.add_argument('--metadata_dir', type=path_or_cloudpath, default=None, help='Path to directory where the metadata will be stored. If not set, infer from data_dir.')
    parser.add_argument('--download_npz', help='If true, also download npz files.', action='store_true', default=False)
    parser.add_argument('--overwrite_metadata', help='If true, force re-download of the metadata files.', action='store_true', default=False)
    parser.add_argument('--skip_bbox_blurring', help='If true, skip bounding box blurring on images while downloading.', action='store_true', default=False)
    parser.add_argument('--processes_count', type=int, required=False, default=16, help='Number of processes for download.')
    parser.add_argument('--thread_count', type=int, required=False, default=128, help='Number of threads for download.')
    
    args = parser.parse_args()

    metadata_dir = args.metadata_dir
    if metadata_dir is None:
        metadata_dir = args.data_dir / 'metadata'

    # Download the metadata files if needed.
    if args.overwrite_metadata or not metadata_dir.exists():
        if metadata_dir.exists():
            print(f'Cleaning up {metadata_dir}')
            shutil.rmtree(metadata_dir)
        metadata_dir.mkdir(parents=True)

        print(f'Downloading metadata to {metadata_dir}...')

        if args.scale != 'xlarge':
            hf_metadata_dir = snapshot_download(
                repo_id=HF_REPO,
                allow_patterns=f'{args.scale}/*.parquet',
                cache_dir=metadata_dir / 'hf',
                repo_type='dataset'
            )
            if args.download_npz:
                print('Downloading npz files')
                snapshot_download(
                    repo_id=HF_REPO,
                    allow_patterns=f'{args.scale}/*.npz',
                    cache_dir=metadata_dir / 'hf',
                    repo_type='dataset'
                )
        else:
            # Slightly different handling for xlarge scale
            hf_metadata_dir = snapshot_download(
                repo_id=HF_REPO,
                allow_patterns=f'{args.scale}/*/*.parquet',
                cache_dir=metadata_dir / 'hf',
                repo_type='dataset'
            )
            if args.download_npz:
                print('Downloading npz files')
                npz_hf_metadata_dir = snapshot_download(
                    repo_id=HF_REPO,
                    allow_patterns=f'{args.scale}_npzs/*/*.npz',
                    cache_dir=metadata_dir / 'hf',
                    repo_type='dataset'
                )

        # Create symlinks
        hf_metadata_dir = Path(hf_metadata_dir) / f'{args.scale}'
        for filename in hf_metadata_dir.rglob('*.parquet'):
            link_filename = metadata_dir / filename.name
            true_filename = filename.resolve()
            link_filename.symlink_to(true_filename)

        if args.download_npz:
            if args.scale != 'xlarge':
                for filename in hf_metadata_dir.rglob('*.npz'):
                    link_filename = metadata_dir / filename.name
                    true_filename = filename.resolve()
                    link_filename.symlink_to(true_filename)
            else:
                npz_hf_metadata_dir = Path(npz_hf_metadata_dir) / f'{args.scale}_npzs'
                for filename in npz_hf_metadata_dir.rglob('*.npz'):
                    link_filename = metadata_dir / filename.name
                    true_filename = filename.resolve()
                    link_filename.symlink_to(true_filename)

        print('Done downloading metadata.')
    else:
        print(f'Skipping download of metadata because {metadata_dir} exists. Use --overwrite_metadata to force re-downloading.')

    # Download images.
    shard_dir = args.data_dir / 'shards'
    shard_dir.mkdir(parents=True, exist_ok=True)
    print(f'Downloading images to {shard_dir}')

    bbox_col = None if args.skip_bbox_blurring else 'face_bboxes'

    img2dataset.download(
        url_list=str(metadata_dir),
        image_size=512,
        output_folder=str(shard_dir),
        processes_count=args.processes_count,
        thread_count=args.thread_count,
        resize_mode='keep_ratio_largest',
        resize_only_if_bigger=True,
        output_format='webdataset',
        input_format='parquet',
        url_col='url',
        caption_col='text',
        bbox_col=bbox_col,
        save_additional_columns=['uid'],
        number_sample_per_shard=10000,
        oom_shard_count=8
    )

    print('Done!')
    
