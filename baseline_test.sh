python baselines.py --metadata_dir testing/metadata --save_path testing/out/no_filter.npy --name no_filter
python baselines.py --metadata_dir testing/metadata --save_path testing/out/basic_filter.npy --name basic_filter
python baselines.py --metadata_dir testing/metadata --save_path testing/out/laion.npy --name laion
python baselines.py --metadata_dir testing/metadata --save_path testing/out/clip_score_l14_30_percent.npy --name clip_score_l14_30_percent
python baselines.py --metadata_dir testing/metadata --save_path testing/out/clip_score_b32_30_percent.npy --name clip_score_b32_30_percent
python baselines.py --metadata_dir testing/metadata --save_path testing/out/clip_score_l14_90_percent.npy --name clip_score_l14_percent --percentage 0.9
python baselines.py --metadata_dir testing/metadata --save_path testing/out/clip_score_b32_90_percent.npy --name clip_score_b32_percent --percentage 0.9
python baselines.py --metadata_dir testing/metadata --save_path testing/out/clip_score_l14_30_threhsold.npy --name clip_score_l14_threshold --threshold 0.3
python baselines.py --metadata_dir testing/metadata --save_path testing/out/clip_score_b32_30_threshold.npy --name clip_score_b32_threshold --threshold 0.3
