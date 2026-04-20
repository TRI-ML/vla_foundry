#!/usr/bin/env python3
"""
Convert parquet files from S3 into tar shards with JSON files.
Each parquet row becomes a JSON file with UUID filename.
"""

import argparse
import io
import json
import os
import tarfile
import tempfile
import uuid
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow.parquet as pq
import ray

from vla_foundry.aws.s3_io import list_objects
from vla_foundry.aws.s3_path import S3Path
from vla_foundry.aws.s3_utils import create_s3_client


def list_parquet_files(s3_input_path):
    """List all parquet files in S3 path."""
    parsed = S3Path(s3_path=s3_input_path)
    files = list_objects(parsed.bucket, parsed.key, suffix_filter=".parquet")
    return files, parsed.bucket


@ray.remote
def get_parquet_row_count(parquet_file, bucket):
    """Get row count from parquet file without reading full data."""
    try:
        s3_url = f"s3://{bucket}/{parquet_file}"
        with fsspec.open(s3_url, "rb") as f:
            parquet_file_obj = pq.ParquetFile(f)
            row_count = parquet_file_obj.metadata.num_rows

        print(f"File {parquet_file}: {row_count} rows")
        return row_count

    except Exception as e:
        print(f"Error reading metadata for {parquet_file}: {e}")
        return 0


def compute_parquet_shard_plan(parquet_files, row_counts, samples_per_shard):
    """Compute how many complete shards each parquet file will create."""
    parquet_plan = []
    current_shard_idx = 0

    for parquet_file, row_count in zip(parquet_files, row_counts, strict=False):
        # Only create complete shards, discard remainder
        num_complete_shards = row_count // samples_per_shard
        discarded_samples = row_count % samples_per_shard

        parquet_plan.append(
            {
                "parquet_file": parquet_file,
                "row_count": row_count,
                "start_shard_idx": current_shard_idx,
                "num_complete_shards": num_complete_shards,
                "discarded_samples": discarded_samples,
            }
        )

        current_shard_idx += num_complete_shards

    return parquet_plan


@ray.remote
def process_parquet_to_shards(parquet_plan, bucket, s3_output_path, tmp_dir, samples_per_shard):
    print("parquet_plan", parquet_plan)
    """Process one parquet file and create only complete tar shards from it."""
    s3_client = create_s3_client()
    parquet_file = parquet_plan["parquet_file"]
    start_shard_idx = parquet_plan["start_shard_idx"]
    num_complete_shards = parquet_plan["num_complete_shards"]
    discarded_samples = parquet_plan["discarded_samples"]

    # Read the entire parquet file once
    s3_url = f"s3://{bucket}/{parquet_file}"
    with fsspec.open(s3_url, "rb") as f:
        table = pq.read_table(f, columns=["text"])
        df = table.to_pandas()

    results = []

    if discarded_samples > 0:
        print(f"Discarding {discarded_samples} samples from {parquet_file}")

    # Ensure tmp_dir exists before creating subdirectory
    if tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmpdir:
        # Create only complete shards (discard remainder)
        for shard_offset in range(num_complete_shards):
            shard_idx = start_shard_idx + shard_offset
            start_row = shard_offset * samples_per_shard
            end_row = start_row + samples_per_shard

            tar_filename = f"shard_{shard_idx:06d}.tar"
            tar_path = Path(tmpdir) / tar_filename

            sample_count = 0
            with tarfile.open(tar_path, "w") as tar:
                for _, row in df.iloc[start_row:end_row].iterrows():
                    if "text" in row and pd.notna(row["text"]) and row["text"].strip():
                        file_uuid = str(uuid.uuid4())
                        json_content = json.dumps({"text": row["text"]}, ensure_ascii=False)
                        json_bytes = json_content.encode("utf-8")

                        tarinfo = tarfile.TarInfo(name=f"{file_uuid}.json")
                        tarinfo.size = len(json_bytes)
                        tar.addfile(tarinfo, io.BytesIO(json_bytes))
                        sample_count += 1

            # Upload to S3
            _parsed = S3Path(s3_path=s3_output_path)
            bucket_name, s3_key = _parsed.bucket, _parsed.key
            full_s3_key = f"{s3_key.rstrip('/')}/{tar_filename}"
            s3_client.upload_file(str(tar_path), bucket_name, full_s3_key)

            results.append({"shard": tar_filename, "samples": sample_count})
            print(f"Created {tar_filename} with {sample_count} samples")

    return results


def create_shards(s3_input_path, s3_output_path, samples_per_shard, tmp_dir, max_concurrent):
    """Convert parquet files to tar shards, parallelizing by parquet file."""
    parquet_files, bucket = list_parquet_files(s3_input_path)
    print(f"Found {len(parquet_files)} parquet files")

    # Step 1: Get row counts to plan how many shards each parquet will create
    print("Getting row counts...")
    count_futures = [get_parquet_row_count.remote(pf, bucket) for pf in parquet_files]
    row_counts = ray.get(count_futures)
    total_estimated_rows = sum(row_counts)
    print(f"Estimated {total_estimated_rows} total rows across all parquet files")

    # Step 2: Create plan for each parquet file (only complete shards)
    print("Creating parquet processing plan...")
    parquet_plan = compute_parquet_shard_plan(parquet_files, row_counts, samples_per_shard)
    total_shards = sum(p["num_complete_shards"] for p in parquet_plan)
    total_discarded = sum(p["discarded_samples"] for p in parquet_plan)
    print(f"Will create {total_shards} complete shards")
    print(f"Will discard {total_discarded} samples from parquet file endings")

    # Step 3: Process parquet files in parallel batches
    print("Processing parquet files...")
    all_results = []
    for i in range(0, len(parquet_plan), max_concurrent):
        batch_futures = []
        for parquet_info in parquet_plan[i : i + max_concurrent]:
            future = process_parquet_to_shards.remote(parquet_info, bucket, s3_output_path, tmp_dir, samples_per_shard)
            batch_futures.append(future)

        batch_results = ray.get(batch_futures)
        for result_list in batch_results:
            all_results.extend(result_list)

        current_shard_idx = i // max_concurrent + 1
        total_shards = (len(parquet_plan) + max_concurrent - 1) // max_concurrent
        print(f"Completed parquet batch {current_shard_idx}/{total_shards}")

    # Print summary
    total_samples = sum(r["samples"] for r in all_results)
    print(f"Created {len(all_results)} shards with {total_samples} total samples")


def main():
    parser = argparse.ArgumentParser(description="Convert parquet files to tar shards")
    parser.add_argument("--s3_input_path", required=True, help="S3 path with parquet files")
    parser.add_argument("--s3_output_path", required=True, help="S3 output path for shards")
    parser.add_argument("--samples_per_shard", type=int, default=8192, help="Samples per shard")
    parser.add_argument("--tmp_dir", type=str, default="/tmp", help="Temp directory")
    parser.add_argument("--ray_address", type=str, default=None, help="Ray cluster address")
    parser.add_argument("--max_concurrent", type=int, default=None, help="Max concurrent shards (default: auto-detect)")

    args = parser.parse_args()

    # Initialize Ray
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init(address="auto")

    max_concurrent = int(args.max_concurrent or ray.cluster_resources().get("CPU", 1))

    try:
        create_shards(args.s3_input_path, args.s3_output_path, args.samples_per_shard, args.tmp_dir, max_concurrent)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
