import os

from baselines.utils import (
    get_threshold,
    save_uids,
    load_uids_with_threshold,
    load_uids_with_basic_filter,
    load_uids_with_text_entity,
)


def function_wrapper(args):
    uids = None
    if args.name == "no_filter":
        uids = load_uids_with_threshold(
            args.metadata_dir,
            "clip_b32_similarity_score",
            float("-inf"),
            args.num_workers,
        )
    elif args.name == "basic_filter":
        uids = load_uids_with_basic_filter(
            args.metadata_dir,
            args.num_workers,
        )
    elif args.name == "text_based":
        uids = load_uids_with_text_entity(
            args.metadata_dir,
            args.num_workers,
        )
    elif args.name == "image_based":
        raise NotImplementedError()
    elif args.name == "laion":
        uids = load_uids_with_threshold(
            args.metadata_dir,
            "clip_b32_similarity_score",
            0.28,
            args.num_workers,
        )
    elif args.name == "clip_score_l14_30_percent":
        threshold = get_threshold(
            args.metadata_dir,
            "clip_l14_similarity_score",
            p=0.3,
            n_workers=args.num_workers,
        )
        uids = load_uids_with_threshold(
            args.metadata_dir,
            "clip_l14_similarity_score",
            threshold,
            args.num_workers,
        )
    elif args.name == "clip_score_b32_30_percent":
        threshold = get_threshold(
            args.metadata_dir,
            "clip_b32_similarity_score",
            p=0.3,
            n_workers=args.num_workers,
        )
        uids = load_uids_with_threshold(
            args.metadata_dir,
            "clip_b32_similarity_score",
            threshold,
            args.num_workers,
        )
    elif args.name == "datacomp_1b":
        raise NotImplementedError()
    elif args.name == "clip_score_l14_percent":
        threshold = get_threshold(
            args.metadata_dir,
            "clip_l14_similarity_score",
            p=args.percentage,
            n_workers=args.num_workers,
        )
        uids = load_uids_with_threshold(
            args.metadata_dir,
            "clip_l14_similarity_score",
            threshold,
            args.num_workers,
        )
    elif args.name == "clip_score_b32_percent":
        threshold = get_threshold(
            args.metadata_dir,
            "clip_b32_similarity_score",
            p=args.percentage,
            n_workers=args.num_workers,
        )
        uids = load_uids_with_threshold(
            args.metadata_dir,
            "clip_b32_similarity_score",
            threshold,
            args.num_workers,
        )
    elif args.name == "clip_score_l14_threshold":
        uids = load_uids_with_threshold(
            args.metadata_dir,
            "clip_l14_similarity_score",
            args.threshold,
            args.num_workers,
        )
    elif args.name == "clip_score_b32_threshold":
        uids = load_uids_with_threshold(
            args.metadata_dir,
            "clip_b32_similarity_score",
            args.threshold,
            args.num_workers,
        )
    else:
        raise ValueError(f"Unknown args.name argument: {args.name}")

    print(f"saving {args.save_path} with {len(uids)} entries")
    save_uids(uids, args.save_path)
