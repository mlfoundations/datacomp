import argparse
import os
from pathlib import Path

from baselines.functions import function_wrapper

NAMED_BASELINES = {
    "no_filter",
    "basic_filter",
    "text_based",
    "image_based",
    "laion",
    "clip_score_l14_30_percent",
    "clip_score_b32_30_percent",
    "datacomp_1b",
}

UNNAMED_BASELINES = {
    "clip_score_l14_percent",
    "clip_score_b32_percent",
    "clip_score_l14_threshold",
    "clip_score_b32_threshold",
}


def check_args(args):
    if args.name not in NAMED_BASELINES or args.name not in UNNAMED_BASELINES:
        raise ValueError(f"--name must be in: {NAMED_BASELINES} or {UNNAMED_BASELINES}")
    if args.name == "clip_score_l14_percent" or args.name == "clip_score_b32_percent":
        if args.percentage is None:
            raise ValueError(
                "--percentage value must be passed for *_percent baselines"
            )
    if (
        args.name == "clip_score_l14_threshold"
        or args.name == "clip_score_b32_threshold"
    ):
        if args.threshold is None:
            raise ValueError(
                "--threshold value must be passed for *_threshold baselines"
            )
    npy_parent = Path(args.save_path).parent
    if not os.path.exists(npy_parent):
        print(f"creating: {npy_parent}")
        os.mkdir(npy_parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        choices=list(NAMED_BASELINES.union(UNNAMED_BASELINES)),
        help="name of the baseline",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="local path to output .npy",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=None,
        help="a threshold to appply to metadata (e.g., keep CLIP score over 0.3)",
    )

    parser.add_argument(
        "--percentage",
        type=float,
        required=False,
        default=None,
        help="a percentage of metadata to keep (e.g., 0.3 for 30\%)",
    )

    parser.add_argument(
        "--metadata_dir",
        type=str,
        required=True,
        help="directory (local or cloud) containing parquet, npz metadata",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=50,
        help="number of workers, generally set to number of cpu cores",
    )

    args = parser.parse_args()

    function_wrapper(args)
