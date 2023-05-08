# Clustering

Generates cluster centroids from the datacomp dataset using k-means clustering. 


## Installing dependencies

We use `faiss-gpu` for k-means clustering. To install, run the following commands:

```bash
conda install -c pytorch faiss-gpu
```

Or check out [faiss-gpu](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

## Run clustering

To run clustering, run the following command:


```
python clustering.py \
        --metadata_dir path/to/metadata \ 
        --save_path path/to/output/centroids \
        --num_clusters 100000 \
        --sample_ratio 0.3 \
        --num_gpus 8 \
        --num_workers 10 \
        --basic_filtering
```

Explanation to several arguments:

- `sample_ratio`: ratio of samples to use in clustering. We do this (for `large` and `xlarge`) because of memory constraint. Default is -1.0 (no sampling)
- `basic_filtering`: whether to apply text-based filtering to the dataset. Default is `False`

