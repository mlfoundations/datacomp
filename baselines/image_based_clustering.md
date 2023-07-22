# Clustering

Generates cluster centroids from the `image-based` baselines using k-means clustering.


## Installing dependencies

We use `faiss-gpu` for k-means clustering. To install, run the following commands:

```bash
conda install -c conda-forge faiss-gpu
```

Or check out [faiss-gpu](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

## Run clustering

To run clustering for the `small` pool, run the following command:


```
python image_based_clustering.py \
        --metadata_dir path/to/metadata \
        --save_path path/to/output/centroids \
        --num_clusters 100000 \
        --sample_ratio -1.0 \
        --num_gpus 8 \
        --num_workers 26 \
```

Explanation to several arguments:

- `sample_ratio`: the ratio of samples to use in clustering. In particular, we sample `sample_ratio` percent embeddings to do clustering due to memory constraint. We use 0.3 for `large` and 0.03 for `xlarge`. Default is -1.0 (no sampling)
- `disable_caption_filtering`: whether to disable caption filtering to the dataset. Default is `False`

On a machine with 8 GPUs and 26 CPUs (there are 26 parquet files for the `small` pool), the clustering process takes about 10 minutes.
