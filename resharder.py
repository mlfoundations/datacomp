#!/usr/bin/env python3

import argparse
import bisect
import copy
import logging
import multiprocessing as mp
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing.managers import AcquirerProxy, NamespaceProxy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import simdjson
import tqdm
import webdataset as wds
from cloudpathlib import CloudPath
from cloudpathlib.enums import FileCacheMode
from img2dataset.blurrer import BoundingBoxBlurrer
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

# we always read and write files exactly once so we can use the strictest caching policy
os.environ["CLOUPATHLIB_FILE_CACHE_MODE"] = FileCacheMode.close_file.name

Pipe = wds.writer.gopen.Pipe
Pathy = Union[Path, CloudPath]


class ColoredConsoleHandler(logging.Handler):
    # TODO: Abstract ANSI color escapes
    def __init__(self, sub_handler=None):
        super().__init__()
        self.sub_handler = (
            logging.StreamHandler() if sub_handler is None else sub_handler
        )

    def emit(self, record):
        # Need to make a actual copy of the record
        # to prevent altering the message for other loggers
        myrecord = copy.copy(record)
        levelno = myrecord.levelno

        # NOTSET and anything else
        color = "\x1b[0m"  # normal
        tag = "NOTSET"

        if levelno >= logging.FATAL:
            color = "\x1b[31m"  # red
            tag = "FATAL"
        elif levelno >= logging.ERROR:
            color = "\x1b[31m"  # red
            tag = "ERROR"
        elif levelno >= logging.WARNING:
            color = "\x1b[33m"  # yellow
            tag = "WARN"
        elif levelno >= logging.INFO:
            color = "\x1b[32m"  # green
            tag = "INFO"
        elif levelno >= logging.DEBUG:
            color = "\x1b[35m"  # pink
            tag = "DEBUG"

        myrecord.msg = f"{color}[{tag}]\x1b[0m {myrecord.msg}"
        self.sub_handler.emit(myrecord)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class MultiProcessingHandler(logging.Handler):
    def __init__(self, name, queue):
        super().__init__()
        self.name = name
        self.queue = queue

    def _format_record(self, record):
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            self.format(record)
            record.exc_info = None

        record.msg = f"[{self.name}] {record.msg}"

        return record

    def emit(self, record):
        record = self._format_record(record)
        self.queue.put_nowait(record)


def setup_process_logging(log_queue, worker_id):
    logger = logging.getLogger("resharder")
    for handler in logger.handlers:
        logger.removeHandler(handler)

    logger.addHandler(MultiProcessingHandler(f"worker {worker_id:03d}", log_queue))
    return logger


logger = logging.getLogger("resharder")
logger.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
logger.addHandler(ColoredConsoleHandler(log_handler))
log_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))


# Monkey-patch webdataset to support S3 via aws s3


class ResharderPipe(Pipe):
    def wait_for_child(self):
        self.status = self.proc.wait()
        if self.proc.stderr:
            stderr = self.proc.stderr.read().decode()
            # Don't pass KeyboardInterrupt through
            if stderr and not stderr.endswith("\nKeyboardInterrupt\n"):
                msg = stderr.rstrip("\n")
                logger.error(f"ResharderPipe captured error: {msg}")

        if self.status not in self.ignore_status and not self.ignore_errors:
            logger.error(
                f"ResharderPipe {self.args}: exit {self.status} (read) {wds.writer.gopen.info}"
            )

    def __del__(self):
        self.stream.close()
        self.proc.wait(self.timeout)


def gopen_aws(url, mode="rb", bufsize=8192):
    """Open a URL with `aws s3`.
    :param url: url (usually, s3:// etc.)
    :param mode: file mode
    :param bufsize: buffer size
    """
    # TODO not sure about ignore_status
    if mode[0] == "r":
        cmd = f"aws s3 cp '{url}' -"
        return ResharderPipe(
            cmd,
            mode=mode,
            shell=True,
            bufsize=bufsize,
            ignore_status=[141, 23],
            stderr=subprocess.PIPE,
        )
    elif mode[0] == "w":
        cmd = f"aws s3 cp - '{url}'"
        return ResharderPipe(
            cmd,
            mode=mode,
            shell=True,
            bufsize=bufsize,
            ignore_status=[141, 26],
            stderr=subprocess.PIPE,
        )
    else:
        raise ValueError(f"{mode}: unknown mode")


wds.gopen_schemes.setdefault("s3", gopen_aws)


class ShardWriter:
    """Like TarWriter but splits into multiple shards."""

    def __init__(
        self,
        namer: Callable,
        maxcount: int = 100000,
        maxsize: float = 3e9,
        post: Optional[Callable] = None,
        start_shard: int = 0,
        logger: Optional[logging.Logger] = None,
        **kw,
    ):
        """Create a ShardWriter.
        :param namer: function mapping shard number to output file name
        :param maxcount: maximum number of records per shard (Default value = 100000)
        :param maxsize: maximum size of each shard (Default value = 3e9)
        :param kw: other options passed to TarWriter
        """
        self.verbose = 1
        self.kw = kw
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.post = post

        self.tarstream = None
        self.shard = start_shard
        self.namer = namer
        self.logger = logger
        self.total = 0
        self.count = 0
        self.size = 0
        self.fname = None

    def next_stream(self):
        """Close the current stream and move to the next."""
        self.finish()
        self.fname = self.namer(self.shard)

        self.shard += 1

        self.tarstream = wds.TarWriter(self.fname, **self.kw)

        self.count = 0
        self.size = 0

    def write(self, obj):
        """Write a sample.
        :param obj: sample to be written
        """
        if (
            self.tarstream is None
            or self.count >= self.maxcount
            or self.size >= self.maxsize
        ):
            self.next_stream()

        try:
            size = self.tarstream.write(obj)
            self.count += 1
            self.total += 1
            self.size += size

        except Exception:
            logger.error(traceback.format_exc())

            # outrageous hack to ensure we don't write more to the broken pipe
            self.tarstream.tarstream.fileobj.closed = True
            self.tarstream = None
            self.next_stream()

    def finish(self):
        """Finish all writing (use close instead)."""
        if self.tarstream is not None:
            try:
                self.tarstream.close()

            except Exception:
                logger.error(traceback.format_exc())

            assert self.fname is not None
            if callable(self.post):
                self.post(fname=self.fname, count=self.count, size=self.size)
            self.tarstream = None

            self.logger.debug(
                f"wrote {self.fname} {self.size / 1e9:.1f} GB, {self.count}/{self.total}"
            )

    def close(self):
        """Close the stream."""
        self.finish()
        del self.tarstream
        del self.shard
        del self.count
        del self.size

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, *args, **kw):
        """Exit context."""
        self.close()


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.
    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()

        if current_sample is None or prefix != current_sample["__key__"]:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])

        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if suffix in current_sample:
            if handler is not None:
                handler(
                    ValueError(
                        f"{fname}: duplicate file name in tar file {suffix} {set(current_sample.keys())}"
                    )
                )
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])

        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value

    if valid_sample(current_sample):
        yield current_sample


def tarfile_samples_nothrow(src, handler):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    try:
        streams = url_opener(src, handler=handler)
        files = tar_file_expander(streams, handler=handler)
        samples = group_by_keys_nothrow(files, handler=handler)

    except Exception as exn:
        exn.args = exn.args + (src)
        handler(exn)
        return []

    return samples


tarfile_to_samples_nothrow = wds.filters.pipelinefilter(tarfile_samples_nothrow)


@dataclass(frozen=True, slots=True)
class Shard:
    shard_id: int
    data_start: int
    size: int


@dataclass
class WorkerTask:
    worker_id: int
    shards: List[Shard]
    parquets: Optional[List[str]]


u16 = np.dtype("u8,u8")


def ceildiv(a, b):
    return -(-a // b)


def path_or_cloudpath(s: str) -> Pathy:
    if re.match(r"^\w+://", s):
        return CloudPath(s.rstrip("/"))
    return Path(s)


def make_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=path_or_cloudpath,
        required=True,
        help="input directory containing a webdataset",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=path_or_cloudpath,
        required=True,
        help="output directory",
    )
    parser.add_argument(
        "-s",
        "--subset-file",
        type=path_or_cloudpath,
        required=True,
        help="subset file, either a NumPy or memmap array of 128 bit hashes",
    )
    parser.add_argument(
        "-n",
        "--num-shards",
        type=int,
        help="number of shards to process (beware of off-by-ones)",
    )
    parser.add_argument(
        "--first-shard",
        type=int,
        default=0,
        help="index of first shard to process",
    )
    parser.add_argument(
        "-j",
        "--num-workers",
        default=mp.cpu_count(),
        type=int,
        help="number of workers to use",
    )
    parser.add_argument(
        "--shard-size",
        default=10000,
        type=int,
        help="maximum number of examples per output shard",
    )
    parser.add_argument(
        "--shard-format",
        default="{:08d}.tar",
        type=str,
        help="format for each input shard in str.format syntax",
    )
    parser.add_argument(
        "--output-shard-format",
        default="{:08d}.tar",
        type=str,
        help="format for each output shard in str.format syntax",
    )
    parser.add_argument(
        "--shard-stats-format",
        default="{:08d}_stats.json",
        type=str,
        help="format for each input shard stats file in str.format syntax",
    )
    parser.add_argument(
        "--output-shard-stats-format",
        default="{:08d}_stats.json",
        type=str,
        help="format for each output shard stats file in str.format syntax",
    )
    parser.add_argument(
        "--shard-table",
        default="sizes.json",
        type=path_or_cloudpath,
        help="JSON file recording input shard sizes relative to INPUT_DIR",
    )
    parser.add_argument(
        "--write-shard-table",
        action="store_true",
        help="write shard table to output_dir if it does not exist",
    )
    parser.add_argument(
        "--shuffle-bufsize", default=0, type=int, help="buffer size for shuffling"
    )
    parser.add_argument(
        "--blur-metadata-map",
        type=path_or_cloudpath,
        default=None,
        help="Map file from shards to parquets for blurring.",
    )
    parser.add_argument(
        "--apply-blur",
        action="store_true",
        help="Apply blurring to images and re-encode them",
    )
    parser.add_argument(
        "--inject-blur-metadata",
        action="store_true",
        help="Add blur bounding boxes to the json field of the output examples",
    )
    parser.add_argument(
        "--reencode-webp-quality",
        type=str,
        default=100,
        help="Quality for re-encoding images if necessary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="do not make any changes to the output directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing files in the output directory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="append_const",
        const=1,
        help="decrease the logging level",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="append_const",
        const=1,
        help="increase the logging level",
    )
    return parser


parser = make_argparser()


def guess_num_shards(
    *,
    input_dir: Pathy,
    first_shard: int = parser.get_default("first_shard"),
    shard_format: str = parser.get_default("shard_format"),
    **_,
):
    n = 1

    def test_size(i):
        shard = input_dir / shard_format.format(first_shard + i - 1)
        return shard.exists()

    for _ in range(40):
        if not test_size(n):
            break
        n *= 2
    else:
        raise RuntimeError(f"Found too many shards (at least {n})")

    if n == 1:
        raise RuntimeError("Did not find any shards")

    n = (
        n // 2
        + bisect.bisect_right(range(n // 2, n), False, key=lambda i: not test_size(i))
        - 1
    )

    return n


def load_shard_size(args):
    shard_id, input_dir, shard_format, shard_stats_format = args
    size_path = input_dir / shard_stats_format.format(shard_id)
    shard_name = shard_format.format(shard_id)
    shard_path = input_dir / shard_name
    size = None
    if size_path.exists() and shard_path.exists():
        with size_path.open("r") as f:
            size = int(simdjson.Parser().parse(f.read()).get("successes"))
    return shard_name, size


def load_shard_metadata(
    *,
    input_dir: Pathy,
    num_shards: int = parser.get_default("num_shards"),
    first_shard: int = parser.get_default("first_shard"),
    shard_format: str = parser.get_default("shard_format"),
    shard_stats_format: str = parser.get_default("shard_stats_format"),
    shard_table: Pathy = parser.get_default("shard_table"),
    write_shard_table: bool = parser.get_default("write_shard_table"),
    num_workers: int = parser.get_default("num_workers"),
    **_,
):
    shards = []
    offset = 0
    parser = simdjson.Parser()

    table = {}
    shard_table_path = input_dir / shard_table
    if shard_table_path.exists():
        logger.info(f"loading shard table {shard_table_path}")
        with open(shard_table_path, "rb") as f:
            try:
                table = simdjson.load(f)
            except ValueError as e:
                logger.error(f"shard table parsing error: {e.args[0]}")
            logger.info(f"shard table has size {len(table)}")

    if not num_shards:
        num_shards = guess_num_shards(
            input_dir=input_dir,
            first_shard=first_shard,
            shard_format=shard_format,
        )
        logger.info(f"binary search found {num_shards} potential shards")

    shard_ids = range(first_shard, first_shard + num_shards)

    with mp.Pool(num_workers) as pool:
        size_iter = pool.imap(
            load_shard_size,
            (
                (
                    shard_id,
                    input_dir,
                    shard_format,
                    shard_stats_format,
                )
                for shard_id in tqdm.tqdm(shard_ids, dynamic_ncols=True, smoothing=0)
                if shard_format.format(shard_id) not in table
            ),
            chunksize=16,
        )

        for shard_name, size in size_iter:
            if size is not None:
                table[shard_name] = size

    missing_shards = 0
    for shard_id in shard_ids:
        shard_name = shard_format.format(shard_id)

        if shard_name in table:
            size = table[shard_name]
            shards.append(Shard(shard_id, offset, size))
            offset += size
        else:
            logger.debug(f"missing shard {shard_name}")
            missing_shards += 1

    if missing_shards > 0:
        logger.warning(
            f"{missing_shards} shards were missing; "
            "set log level to DEBUG to see list"
        )

    total_data = shards[-1].data_start + shards[-1].size
    logger.info(f"found a total of {len(shards)} shards with {total_data} examples")

    if write_shard_table and not shard_table_path.exists():
        logger.info("writing shard table")
        with shard_table_path.open("w") as f:
            simdjson.dump(table, f)

    return shards, total_data


def load_subset(*, subset_file: Path, **_):
    assert not isinstance(subset_file, CloudPath)

    with open(subset_file, "rb") as f:
        # Detect the NumPy format magic string
        if f.read(6) == b"\x93NUMPY":
            subset = np.load(subset_file, mmap_mode="r")
            assert subset.dtype == u16

        else:
            subset = np.memmap(subset_file, u16, mode="r+")

    return subset


def load_parquet_metadata(
    shards: List[Shard],
    /,
    blur_metadata_map: Optional[Pathy] = parser.get_default("blur_metadata_map"),
    shard_format: str = parser.get_default("shard_format"),
    input_dir: Optional[Pathy] = None,
    **_,
):
    if blur_metadata_map is None:
        return None

    with blur_metadata_map.open("r") as f:
        parquets = simdjson.load(f)

    parquet_table = {}

    # invert the parquet → shard multi-map
    for pq in parquets.values():
        for shard in pq["shards"]:
            shard_path = path_or_cloudpath(shard)
            if input_dir is not None and shard_path.parent != input_dir:
                continue
            parquet_table[shard_path.name] = pq["parquet"]

    parquet_list = []
    missing_parquets = 0
    for shard in shards:
        shard_name = shard_format.format(shard.shard_id)
        parquet_list.append(parquet_table.get(shard_name))
        if parquet_list[-1] is None:
            logger.debug(f"could not find parquet for shard {shard_name}")
            missing_parquets += 1

    if missing_parquets > 0:
        logger.warning(
            f"could not find parquets for {missing_parquets} shards; "
            "set log level to DEBUG to see list"
        )

    return parquet_list


def plan_tasks(shards: List[Shard], parquets: Optional[List[str]] = None, /, **args):
    num_workers = args["num_workers"]
    worker_tasks = []
    total_data = shards[-1].data_start + shards[-1].size

    # evenly distribute data to workers
    data_starts = [shard.data_start for shard in shards]
    shard_chunks = [
        np.searchsorted(data_starts, i, side="left")
        for i in range(0, total_data, -(-total_data // num_workers))
    ]
    shard_chunks.append(len(shards))

    for worker_id, (shard_start, shard_end) in enumerate(
        zip(shard_chunks, shard_chunks[1:])
    ):
        if shard_start == shard_end:
            continue
        first_shard, last_shard = shards[shard_start], shards[shard_end - 1]

        first_index = first_shard.data_start
        last_index = last_shard.data_start + last_shard.size - 1

        worker_parquets = (
            parquets[shard_start:shard_end] if parquets is not None else None
        )

        logger.debug(
            f"worker {worker_id:03d} will process shards {shard_start} to {shard_end-1}"
        )
        worker_tasks.append(
            WorkerTask(worker_id, shards[shard_start:shard_end], worker_parquets)
        )

    return worker_tasks


def blur_image(
    blurrer: BoundingBoxBlurrer,
    jpg: bytes,
    blur_bboxes,
    reencode_webp_quality: int = parser.get_default("reencode_webp_quality"),
):
    img_buf = np.frombuffer(jpg, np.uint8)
    decoded = cv2.imdecode(img_buf, cv2.IMREAD_UNCHANGED)
    blurred = blurrer(decoded, blur_bboxes)
    encoded = cv2.imencode(
        ".webp",
        blurred,
        params=[int(cv2.IMWRITE_WEBP_QUALITY), reencode_webp_quality],
    )[1].tobytes()
    return encoded


def load_blur_bboxes(f):
    table = pq.read_table(f, columns=["uid", "face_bboxes"])
    table = table.sort_by("uid")
    uids = pc.ascii_lpad(table[0], 0x20, "0")

    uh = pc.cast(
        pc.binary_join_element_wise(
            "0x", pc.utf8_slice_codeunits(uids, 0x00, 0x10), ""
        ),
        pa.uint64(),
    ).to_numpy()

    lh = pc.cast(
        pc.binary_join_element_wise(
            "0x", pc.utf8_slice_codeunits(uids, 0x10, 0x20), ""
        ),
        pa.uint64(),
    ).to_numpy()

    return np.core.records.fromarrays([uh, lh]), table[1]


def copy_worker(
    task: WorkerTask,
    state: NamespaceProxy,
    lock: AcquirerProxy,
    log_queue,
    *,
    input_dir: Pathy,
    output_dir: Pathy,
    subset_file: Path,
    shard_format: str = parser.get_default("shard_format"),
    output_shard_format: str = parser.get_default("output_shard_format"),
    output_shard_stats_format: str = parser.get_default("output_shard_stats_format"),
    shard_size: int = parser.get_default("shard_size"),
    shuffle_bufsize: int = parser.get_default("shuffle_bufsize"),
    reencode_webp_quality: int = parser.get_default("reencode_webp_quality"),
    apply_blur: bool = parser.get_default("apply_blur"),
    inject_blur_metadata: bool = parser.get_default("inject_blur_metadata"),
    dry_run: bool = parser.get_default("dry_run"),
    **_,
):
    logger = setup_process_logging(log_queue, task.worker_id)

    def log_and_continue(exn):
        logger.error(f"webdataset error: {repr(exn)}")
        return True

    subset = load_subset(subset_file=subset_file)
    ds = wds.DataPipeline(
        wds.SimpleShardList(
            [
                str(input_dir / shard_format.format(shard.shard_id))
                for shard in task.shards
            ]
        ),
        tarfile_to_samples_nothrow(handler=log_and_continue),
    )

    # create shard_name → parquet_name mapping
    assert task.parquets is None or len(task.shards) == len(task.parquets)
    parquet_table = (
        {
            shard_format.format(shard.shard_id): parquet
            for shard, parquet in zip(task.shards, task.parquets)
        }
        if task.parquets is not None
        else {}
    )

    @lru_cache(1)
    def load_parquet(fname):
        try:
            logger.debug(f"loading parquet {fname}")
            with path_or_cloudpath(fname).open("rb") as f:
                return load_blur_bboxes(f)
        except FileNotFoundError:
            return None

    def get_blur_bboxes_for_img(url, uid):
        fname = parquet_table.get(path_or_cloudpath(url).name)
        if fname is not None:
            parquet = load_parquet(fname)
            if parquet is None:
                logger.error(f"failed to find parquet for {url}")

            uids, bboxes = parquet
            i = np.searchsorted(uids, uid)
            if uids[i] != uid:
                logger.error(
                    f"failed to find blur bboxes for {url}, {uid[0]:016x}{uid[1]:016x}"
                )
                return

            return bboxes[i].as_py()

    output_shard_index = None

    def output_shard_namer(_shard):
        nonlocal output_shard_index
        with lock:
            output_shard_index = state.output_shard_count
            state.output_shard_count += 1

        return str(output_dir / output_shard_format.format(output_shard_index))

    def output_shard_size_writer(count, **_):
        with (output_dir / output_shard_stats_format.format(output_shard_index)).open(
            "w"
        ) as f:
            simdjson.dump({"successes": count}, f)

    sw = ShardWriter(
        output_shard_namer,
        maxcount=shard_size,
        logger=logger,
        post=output_shard_size_writer,
    )

    sw.verbose = False

    total_data = (
        task.shards[-1].data_start + task.shards[-1].size - task.shards[0].data_start
    )

    processed_count, output_count, blur_count, blur_time = 0, 0, 0, 0

    def subset_iter():
        parser = simdjson.Parser()
        blurrer = BoundingBoxBlurrer()

        def parse_json_safe(s):
            nonlocal parser
            try:
                return parser.parse(s)
            except RuntimeError:
                logger.warning("discarding parser due to dangling reference")
                # throw away the old parser
                parser = simdjson.Parser()
                return parser.parse(s)

        def process_example(d):
            nonlocal processed_count, output_count, blur_count, blur_time

            if "json" not in d:
                logger.error(
                    f"missing json for {d['__url__']}/{d['__key__']}, skipping"
                )
                return

            json_parsed = parse_json_safe(d["json"])
            key_str = json_parsed.get("uid")
            # TODO: is this really the best way to get a u16 scalar?
            key_u16 = np.array([divmod(int(key_str, 16), 2**64)], u16)[0]

            a = np.searchsorted(subset, key_u16, "left")
            b = np.searchsorted(subset, key_u16, "right")
            count = b - a

            if task.parquets and count > 0:
                blur_bboxes = get_blur_bboxes_for_img(d["__url__"], key_u16)
                if blur_bboxes is not None and len(blur_bboxes) > 0:
                    if apply_blur:
                        blur_start_time = time.perf_counter()
                        d["webp"] = blur_image(
                            blurrer, d["jpg"], blur_bboxes, reencode_webp_quality
                        )
                        del d["jpg"]  # Remove jpg version of image
                        blur_count += 1
                        blur_time += time.perf_counter() - blur_start_time

                    if inject_blur_metadata:
                        json = json_parsed.as_dict()
                        json["face_bboxes"] = list(map(list, blur_bboxes))
                        d["json"] = simdjson.dumps(json).encode()

            for j in range(count):
                if not dry_run:
                    yield {**d, "__key__": f"{key_str}-{j}"}

                output_count += 1

            processed_count += 1

            if processed_count % 1000 == 0:
                log_queue.put_nowait(1000)

            del json_parsed

        for input_data in ds:
            try:
                for output_data in process_example(input_data):
                    yield output_data
            except Exception:
                logger.error(traceback.format_exc())

        log_queue.put_nowait(processed_count % 1000)

    it = subset_iter()
    if shuffle_bufsize > 0:
        it = wds.filters._shuffle(it, shuffle_bufsize, shuffle_bufsize)

    try:
        for d in it:
            try:
                sw.write(d)
            except Exception:
                logger.error(traceback.format_exc())

        try:
            sw.close()
        except Exception:
            logger.error(traceback.format_exc())

        if processed_count != total_data:
            logger.error(f"expected {total_data} samples but found {processed_count}")

        with lock:
            state.worker_success += 1

    except KeyboardInterrupt:
        logger.fatal("Caught KeyboardInterrupt, exiting...")

    finally:
        with lock:
            state.processed_count += processed_count
            state.output_count += output_count
            state.blur_count += blur_count
            state.blur_time += blur_time


def logging_handler(total_data, log_queue):
    bar = tqdm.tqdm(total=total_data, dynamic_ncols=True, smoothing=0)

    # this feels a bit ad-hoc
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(log_handler.formatter)
    handler = ColoredConsoleHandler(tqdm_handler)

    while True:
        try:
            message = log_queue.get(timeout=1)
            if message is None:
                break

            if isinstance(message, int):
                bar.update(message)

            if isinstance(message, logging.LogRecord):
                handler.emit(message)

        except queue.Empty:
            pass

        except:
            traceback.print_exc(file=sys.stderr)
            raise

    bar.close()


def do_tasks(worker_tasks, args):
    manager = mp.Manager()

    state = manager.Namespace()
    state.processed_count = 0
    state.output_count = 0
    state.blur_count = 0
    state.blur_time = 0
    state.output_shard_count = 0
    state.worker_success = 0

    lock = manager.Lock()
    log_queue = manager.Queue()

    # not very elegant
    last_shard = worker_tasks[-1].shards[-1]
    total_data = last_shard.data_start + last_shard.size

    logging_thread = threading.Thread(
        target=logging_handler, args=(total_data, log_queue)
    )

    processes = [
        mp.Process(target=copy_worker, args=(task, state, lock, log_queue), kwargs=args)
        for task in worker_tasks
    ]

    logging_thread.start()
    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        # workers will also receive a KeyboardInterrupt
        # so wait for them to terminate on their own
        for p in processes:
            p.join()

    finally:
        # send the sentinel value to the thread to tell it to exit
        log_queue.put_nowait(None)
        logging_thread.join()

    if state.worker_success != len(worker_tasks):
        logger.error(f"{len(worker_tasks) - state.worker_success} workers failed")

    return state


def rmtree_contents(path: Pathy, /, overwrite, num_workers, **_):
    files_exist = any(path.iterdir())
    if not overwrite and files_exist:
        logger.fatal(
            "refusing to overwrite non-empty directory; "
            "skip this check by passing --overwrite"
        )
        sys.exit(1)

    def remove_file(path):
        if path.is_file():
            path.unlink()

    if files_exist:
        with mp.Pool(num_workers) as pool:
            pool.imap(remove_file, path.iterdir(), chunksize=16)


def postprocess_output(*, output_dir, shard_format, **_):
    logger.info("postprocessing output shards")
    for i, shard in enumerate(sorted(output_dir.iterdir())):
        shard.rename(output_dir / shard_format.format(i))


def set_loglevel(logger, /, verbose, quiet, **_):
    verbose = 0 if verbose is None else sum(verbose)
    quiet = 0 if quiet is None else sum(quiet)
    log_levels = [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]
    logger.setLevel(log_levels[max(min(1 - verbose + quiet, len(log_levels)), 0)])


def make_memory_tmpfile():
    shm = Path("/dev/shm")
    # file is about to be memory-mapped so using a tmpfs
    # saves us a copy if it is not local to begin with
    return tempfile.NamedTemporaryFile(
        "w+b", prefix="resharder-", **({"dir": shm} if shm.exists() else {})
    )


def main(args):
    set_loglevel(logger, **vars(args))

    logger.info("loading shard metadata")
    shards, total_data = load_shard_metadata(**vars(args))
    if len(shards) < args.num_workers:
        args.num_workers = len(shards)

    logger.info("deleting files from output directory")
    rmtree_contents(args.output_dir, **vars(args))

    if args.apply_blur and not args.blur_metadata_map:
        logger.fatal("need to pass --blur-metadata-map to use --apply-blur")

    if args.inject_blur_metadata and not args.blur_metadata_map:
        logger.fatal("need to pass --blur-metadata-map to use --inject-blur-metadata")

    # If blur is needed, retrieve json with metadata parquet locations.
    if args.blur_metadata_map is not None:
        logger.info("loading parquet metadata")
        parquets = load_parquet_metadata(shards, **vars(args))
    else:
        parquets = None

    with make_memory_tmpfile() as f:
        if isinstance(args.subset_file, CloudPath):
            with args.subset_file.open("rb") as sf:
                logger.info("copying remote subset file to local machine")
                shutil.copyfileobj(sf, f)
                f.seek(0)

            args.subset_file = Path(f.name)

        if not args.dry_run:
            with args.subset_file.open("rb") as sf:
                logger.info("copying the subset file to the output directory")
                output_filename = args.output_dir / "sample_ids.npy"

                with output_filename.open("wb") as of:
                    shutil.copyfileobj(sf, of)

        subset = load_subset(**vars(args))
        logger.info(f"selecting a subset of {len(subset)} examples")

        worker_tasks = plan_tasks(shards, parquets, **vars(args))

        logger.info("starting workers...")
        start_time = time.perf_counter()
        state = do_tasks(worker_tasks, vars(args))
        elapsed_time = time.perf_counter() - start_time

        logger.info(
            f"processed {state.processed_count} images in {elapsed_time:.3f}s ({state.processed_count/elapsed_time:.2f} images/sec)"
        )
        if state.processed_count != total_data:
            logger.error(
                f"expected {total_data} samples but found {state.processed_count}"
            )

        logger.info(f"output {state.output_count} images")
        if state.output_count != len(subset):
            logger.warning(
                f"{len(subset) - state.output_count} images in the subset were not found in the input!"
            )

        logger.info(f"wrote {state.output_shard_count} output shards")

        if state.blur_count > 0:
            logger.info(f"applied blur to {state.blur_count} images")
            blur_percent = state.blur_time / (args.num_workers * elapsed_time) * 100
            logger.info(
                f"spent {state.blur_time:.3f} worker seconds ({blur_percent:0.1f}% of total) blurring images"
            )

        if not args.dry_run:
            with (args.output_dir / "meta.json").open("w") as f:
                simdjson.dump(
                    {
                        **{k: str(v) for k, v in vars(args).items()},
                        **vars(state._getvalue()),
                        "cwd": str(Path.cwd()),
                    },
                    f,
                )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
