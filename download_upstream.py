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
    parser.add_argument('--skip_shards', help='If true, only download metadata.', action='store_true', default=False)
    parser.add_argument('--overwrite_metadata', help='If true, force re-download of the metadata files.', action='store_true', default=False)
    parser.add_argument('--skip_bbox_blurring', help='If true, skip bounding box blurring on images while downloading.', action='store_true', default=False)
    parser.add_argument('--processes_count', type=int, required=False, default=16, help='Number of processes for download.')
    parser.add_argument('--thread_count', type=int, required=False, default=128, help='Number of threads for download.')
    parser.add_argument('--image_size', type=int, required=False, default=512, help='Size images need to be downloaded to.')
    parser.add_argument('--resize_mode', type=str, required=False, choices=["no", "border", "keep_ratio", "keep_ratio_largest", "center_crop"], default='keep_ratio_largest', help='Resizing mode used by img2dataset when downloading images.')
    parser.add_argument('--no_resize_only_if_bigger', help='If true, do not resize only if images are bigger than target size.', action='store_true', default=False)
    parser.add_argument('--encode_format', type=str, required=False, choices=["png", "jpg", "webp"], default='jpg', help='Images encoding format.')
    parser.add_argument('--output_format', type=str, required=False, choices=["webdataset", "tfrecord", "parquet", "files"], default='webdataset', help='Output format used by img2dataset when downloading images.')

    args = parser.parse_args()

    metadata_dir = args.metadata_dir
    if metadata_dir is None:
        metadata_dir = args.data_dir / 'metadata'
    if metadata_dir.parts[-1] != args.scale:
        metadata_dir = metadata_dir / f'{args.scale}'

    # Download the metadata files if needed.
    if args.overwrite_metadata or not metadata_dir.exists():
        if metadata_dir.exists():
            print(f'Cleaning up {metadata_dir}')
            shutil.rmtree(metadata_dir)
        metadata_dir.mkdir(parents=True)

        print(f'Downloading metadata to {metadata_dir}...')

        hf_snapshot_args = dict(repo_id=HF_REPO,
                                allow_patterns=f'{args.scale}/*.parquet',
                                local_dir=metadata_dir.parent,
                                cache_dir=metadata_dir.parent / 'hf',
                                local_dir_use_symlinks=False,
                                repo_type='dataset')

        if args.scale == 'xlarge':
            hf_snapshot_args['allow_patterns'] = f'{args.scale}/*/*.parquet'

        snapshot_download(**hf_snapshot_args)
        if args.download_npz:
            hf_snapshot_args['allow_patterns'] = hf_snapshot_args['allow_patterns'].replace('.parquet', '.npz')
            print('\nDownloading npz files')
            snapshot_download(**hf_snapshot_args)

        print('Done downloading metadata.')
    else:
        print(f'Skipping download of metadata because {metadata_dir} exists. Use --overwrite_metadata to force re-downloading.')

    if not args.skip_shards:
        # Download images.
        metadata_dir = metadata_dir / f'{args.scale}'
        shard_dir = args.data_dir / 'shards' / f'{args.scale}'
        shard_dir.mkdir(parents=True, exist_ok=True)
        print(f'Downloading images to {shard_dir}')

        bbox_col = None if args.skip_bbox_blurring else 'face_bboxes'

        img2dataset.download(
            url_list=str(metadata_dir),
            image_size=args.image_size,
            output_folder=str(shard_dir),
            processes_count=args.processes_count,
            thread_count=args.thread_count,
            resize_mode=args.resize_mode,
            resize_only_if_bigger=not args.no_resize_only_if_bigger,
            encode_format=args.encode_format,
            output_format=args.output_format,
            input_format='parquet',
            url_col='url',
            caption_col='text',
            bbox_col=bbox_col,
            save_additional_columns=['uid'],
            number_sample_per_shard=10000,
            oom_shard_count=8
        )
    else:
        print(f'Skipping image data download.')

    print('Done!')
    
