# DataComp

[[ Paper ]](https://arxiv.org/abs/2304.14108) [[ Website ]](http://datacomp.ai/) [[ Blog ]](https://laion.ai/blog/datacomp/)

Welcome to our competition. This repository contains the participant tooling necessary to download data from our pool, train CLIP models, evaluate them on downstream tasks and submit to our leaderboard.

## Overview

DataComp is a competition about designing datasets for pre-training CLIP models.
Instead of iterating on model design and hyperparameter tuning like in traditional benchmarks, in DataComp your task is to curate a multimodal pre-training dataset with image-text pairs that yields high accuracy on downstream tasks.
Model architecture and hyperparameters are fixed allowing participants to innovate on the dataset design.
As part of the benchmark, we provide a large collection of uncurated image-text pairs, crawled from the public internet.

Our benchmark offers two tracks: one where participants must use only samples from the pools we provide (`filtering`), and another where participants can use external data, including samples from our pool (Bring your own data, `BYOD`).

DataComp is structured to accommodate participants with diverse levels of computational resources: each track is broken down into four scales, with varying amounts of compute requirements.

An overview of our benchmark and participant workflow can be found below. For more information, check out our [paper](https://arxiv.org/abs/2304.14108) and [website](http://datacomp.ai/).

<p align="center">
<img src="figs/workflow.png" alt="Participant workflow" width="100%"/>
</p>

## Installing dependencies

Run:

```
bash create_env.sh
```

To activate the environment:

```
conda activate datacomp
```

If using cloud storage services (e.g. AWS S3), you'll need to install additional dependencies (e.g. `pip install 'cloudpathlib[s3]'`).

## Downloading CommonPool

To download, run the following command, replacing `$scale` with the competition scale (i.e. `small`, `medium`, `large` or `xlarge`) and `$data_dir` with the output directory where you want the data to be stored.

```python download_upstream.py --scale $scale --data_dir $data_dir```

There are four scales in our competition:

- `small`: 12.8M pool size, 12.8M examples seen
- `medium`: 128M pool size, 128M examples seen
- `large`: 1.28B pool size, 1.28B examples seen
- `xlarge`: 12.8B pool size, 12.8B examples seen


The script will create two directories inside `$data_dir`: `metadata` and `shards`. 

Along with the images and captions, this script will also download metadata, including `.parquet` files that contain the image urls, captions, and other potentially useful information such as the similarities between the images and captions given by trained OpenAI CLIP models.
If the flag `--download_npz` is used, the script will also download the `.npz` files with features extracted by the trained OpenAI CLIP models for each sample.

We download the image data using [img2dataset](https://github.com/rom1504/img2dataset), which stores it as `.tar` shards with the images and captions to be consumed by [webdataset](https://github.com/webdataset/webdataset/).
Once the download finishes, the data will be available at `$data_dir/shards`.

To download only metadata, use the `--skip_shards` flag.

The disk requirements for each scale are shown below.

|                 | metadata (parquets) | metadata (npzs) | data (tars) |
| :-------------- | :-----------------: | :-------------: | :---------: |
| `small` scale   |        3 GB         |      75GB       |    450 GB   |
| `medium` scale  |       30 GB         |     750GB       |    4.5 TB   |
| `large` scale   |      300 GB         |     7.5TB       |    45 TB    |
| `xlarge` scale  |        3 TB         |      75TB       |    450 TB   |

### Downloading DataComp-1B

The script `download_upstream.py` can be used to download the `DataComp-1B` dataset that we release as our best performing subset of the `xlarge` pool. To download this, use the following command:

```python download_upstream.py --scale datacomp_1b --data_dir $data_dir```

The above command will create the same directory structure under `$data_dir` and can be modified as described above.

### Downloading external data

The script `download_upstream.py` can also be used to download other image-text datasets, using [img2dataset](https://github.com/rom1504/img2dataset).
Given parquet files containing the image urls and captions, you can use this script to download the images, by using the flag `--metadata_dir` to point to the directory where the parquet files are stored.
By default, we also download the parquet files corresponding to the pools we provide, and this metadata is stored in a subfolder of `$data_dir`.

### Optimizing the download

When using img2dataset, there are several ways to optimize the download process such as using multiple nodes in a distributed environment or setting up a DNS resolver to increase the success rate of images being downloaded. See the [img2dataset repository](https://github.com/rom1504/img2dataset) for further instructions on how to optimize the download process, as well as information on potential issues during the download.

## Selecting samples in the filtering track

Before training, you will need to select the subset of samples you wish to use. Given a set of chosen samples, we create new shards with only those samples, which the training code then consumes. For each scale, **models are trained for a fixed number of steps**, regardless of the size of the chosen subset of the provided pool.

Each sample in our pool has a unique identifier, which is present in the metadata parquets, and in the `json` files inside the `.tar` shards.

The format describing the subset of samples should be a numpy array of dtype `numpy.dtype("u8,u8")` (i.e. a [structured array](https://numpy.org/doc/stable/user/basics.rec.html) of pairs of unsigned 64-bit integers), with shape `(subset_size,)`, containing a list of `uid`s (128-bit hashes from the parquet files) in *lexicographic sorted order*, saved to disk in either [`npy` format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html) or [memory-mapped format](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html).

For instance, if you have a list of uids `uids = ['139e4a9b22a614771f06c700a8ebe150', '6e356964a967af455c8016b75d691203']`, you can store them by running the following python code:

```
processed_uids = np.array([(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids], np.dtype("u8,u8"))
processed_uids.sort()
np.save(out_filename, processed_uids)
```

After creating a subset, you may invoke the resharder to build the subset shards in `$output_dir` like so:

```
python resharder.py -i $download_dir -o $output_dir -s $subset_file
```

If desired, the resharder can be run in parallel on multiple nodes. The easiest way to do so is to split the input directory into smaller subfolders with fewer shards, and run separate resharder jobs for each of them, each with to separate output directories.

## Baselines

Here we provide command lines for the main filter baselines found in Table 3 of our paper, along with short descriptions. Each baseline reads the `.parquet` metadata files (and also the `.npz` files when needed) , selects a subset of `uids`, sorts them, and saves them to a `.npy` subset file. This file can then be input to the resharder described above to create a webdataset containing only the selected subset of the pool.

**Note**: the `--num_workers` flag controls the number of metadata files that are read into memory and processed on parallel. It is set by default to the number of cores, but that may be too much for machine with many cores and limited memory. For baselines other than image-filtering, allow at least 256MB of memory per worker.

### No filtering
Here we load all metadata `uids` without any additional filtering.
```
python baselines.py --metadata_dir path/to/metadata --save_path path/to/no_filter.npy --name no_filter
```

### Basic filtering
Simple checks on caption length, english being the detected caption language, image size, and image aspect ratio.
```
python baselines.py --metadata_dir path/to/metadata --save_path path/to/basic_filter.npy --name basic_filter
```

### CLIP score filtering
Retain the top k=0.3 fraction of the pool by L/14 CLIP score.
```
python baselines.py --metadata_dir path/to/metadata --save_path path/to/clip_score_l14_30_percent.npy --name clip_score --arch l14 --fraction 0.3
```

Retain all examples with B/32 CLIP score above 0.25.
```
python baselines.py --metadata_dir path/to/metadata --save_path path/to/clip_score_b32_25_threshold.npy --name clip_score --arch b32 --threshold 0.25
```

### LAION-2B filtering
Reproduces the filtering strategy used to create the LAION-2B dataset: applies a B/32 CLIP score filter on image-text pairs, retaining samples with score above 0.28, and an English filter using the gcld3 model to detect language.
```
python baselines.py --metadata_dir path/to/metadata --save_path path/to/laion.npy --name laion2b
```

### Text-based filtering
A text filter captions that contain words from the ImageNet-21k synsets.
```
python baselines.py --metadata_dir path/to/metadata --save_path path/to/text_based.npy --name text_based
```

### Image-based filtering
A image clustering based method that retains samples whose images have content close to ImageNet-1k training images, as measured by the nearest-neighbor cluster center of the image's L/14 CLIP embedding.

**Note**: this baseline uses GPU resources. By default it will try to use all GPUs. To control which GPUs are used, set the `CUDA_VISIBLE_DEVICES` environment variable.
```
python baselines.py --metadata_dir path/to/metadata --save_path path/to/image_based.npy --name image_based --image_based_scale small --batch_size 512
```

**Note**: this baseline requires pre-computed image cluster centroids which will be downloaded automatically the first time you run it. 
If you want to generate the centroids yourself, please see `baselines/image_based_clustering.md` for instructions.

### Intersection of image-based and CLIP score filtering
Applies both the `CLIP score (L/14) with top 0.3 fraction` filter and an `Image-based` filter. This is our best performing baseline for `medium`, `large`, and `xlarge` scales. We used this strategy at the `xlarge` scale to create the `DataComp-1B` dataset.

**Note**: this baseline uses GPU resources. By default it will try to use all GPUs. To control which GPUs are used, set the `CUDA_VISIBLE_DEVICES` environment variable.
```
python baselines.py --metadata_dir path/to/metadata --save_path path/to/image_based_intersect_clip_score_l14_30_percent.npy --name image_based_intersect_clip_score --image_based_scale small --batch_size 512 --arch l14 --fraction 0.3
```

## Training

To train, run the following command:

```torchrun --nproc_per_node $num_gpus train.py --scale $scale --data_dir $data_dir --output_dir $output_dir --exp_name $exp_name```

We support using multiple different data directories. For instance, if your data is in `/path/to/dir/1` and in `/path/to/dir/2`, you can use the flag `--data_dir=/path/to/dir/1::/path/to/dir/2`.


A sample script for training with SLURM with is provided at `tools/slurm_train.sh`.

#### Hyper-parameters

The hyper-parameters used for training are fixed for a given scale. For the `small` and `medium` scales we use ViT-B/32 models, for the `large` scale, ViT-B/16, and for the `xlarge` scale, ViT-L/14. The number of samples seen during training is determined by the scale, and is equal to the size of the corresponding pool we provide. Additional details on hyper-parameters can be found in the paper.

#### Note on variance across runs

We observed small (but non-zero) variance in performance when changing random seeds, seeing differences in accuracy typically at the range of 0.2 percentage points on ImageNet and up to 0.006 on average.
We also note that some factors can make runs non-deterministic even when setting the same random seed (for example, random network failures when streaming data can cause different batches to be formed when re-running, see also https://pytorch.org/docs/stable/notes/randomness.html).

## Evaluation

### [Optional] Pre-download evaluation datasets

Pre-downloading evaluation datasets is optional if you have a strong Internet connection; by default, the data will be streamed directly from Hugging Face Hub. If you wish to download the data, run the following command, replacing `$download_dir` with your desired download path:

```
python download_evalsets.py $download_dir
```

### Evaluating

To evaluate, run the following command:

```
python evaluate.py  --train_output_dir $train_output_dir/$exp_name
```

If you have already donwloaded the datasets, you can use the flag `--data_dir` to point the code to the path where the data is stored.
By default, the evaluation script outputs to the same directory as `$train_output_dir`. This can be changed with the flag `--output_dir` on the evaluation script.

**Note:** This will not submit to our leaderboard unless you pass the `--submit` flag.

## Submitting

To submit, you'll run the evaluate script with some extra flags.

The submission script will upload files to Hugging Face Hub (like the model checkpoint and the file specifying the sample ids), and you will need a Hugging Face account for that, and a repository where these artifacts will be stored. To do so, follow these steps:

1. Make sure you have `git-lfs` installed (run `git lfs install` if not)
2. Create a Hugging Face account at https://huggingface.co/join.
3. Login to your Hugging Face account: `huggingface-cli login`
4. Create a repository where the data will be stored: `huggingface-cli repo create <REPO_NAME> --type model`.

Once you're ready to submit, run the evaluation script with some extra flags, for example:

```
python evaluate.py --track=filtering --train_output_dir=$train_output_dir --submit --method_name="your_awesome_method_name" --samples=$sample_files --author="your_name" --email=your@email.com --hf_username=$hf_username --hf_repo_name=$hf_repo_name
```

Please note that the name of your method and the authors (and no other information) will be made publicly available in our leaderboard.

**Important:** We highly encourage users to specify the samples used to train the model using the `--samples` flag. This can be either file(s) containing the uids of samples from our pool, and/or other files specifying the urls and captions for images outside our pool. You can specify multiple files using the `::` separator, for instance `--samples=/path/to/sample_ids.npy::/path/to/custom_data.parquet`.
We also highly encourage participants to also upload the checkpoints for their trained models using the `--upload-checkpoint` flag.

## Checkpoints

We release the checkpoints for our main baselines as part of [OpenCLIP](https://github.com/mlfoundations/open_clip). More details can be found at https://github.com/mlfoundations/open_clip/blob/main/docs/datacomp_models.md. 

## Citation

If you found this repository, our paper or data useful, please consider citing:

```
@article{datacomp,
  title={DataComp: In search of the next generation of multimodal datasets},
  author={Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim Entezari, Giannis Daras, Sarah Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga Saukh, Alexander Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont, Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, Ludwig Schmidt},
  journal={arXiv preprint arXiv:2304.14108},
  year={2023}
}
```
