"""Reusable data download/conversion helpers for VLA Foundry tutorials.

Each function downloads a small dataset sample from HuggingFace, converts it into
the local format expected by the training pipeline, and returns the path to the
output manifest or directory.  All functions are idempotent -- they skip work when
the output already exists.

The helpers at the top of this file (write_tar_shard, write_manifest, decode_video_frame_range,
hf_download_with_retries) mirror patterns from the core preprocessing pipeline
(vla_foundry.data.preprocessing.utils, …robotics.converters.lerobot) but avoid pulling
in Ray / S3 / heavy VLA dependencies so the tutorials stay lightweight.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import warnings
from pathlib import Path
from typing import Any

import av
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import zstandard
from huggingface_hub import hf_hub_download
from PIL import Image

# ---------------------------------------------------------------------------
# Shared lightweight helpers
# ---------------------------------------------------------------------------


def write_tar_shard(
    out_dir: str,
    shard_idx: int,
    samples: list[dict[str, bytes]],
) -> dict[str, Any]:
    """Write *samples* into a single WebDataset tar shard.

    Each sample is a ``{filename: payload_bytes}`` dict.  Returns a manifest
    entry ``{"shard": "<stem>", "num_sequences": N}``.

    Mirrors the shard-writing logic in
    ``vla_foundry.data.preprocessing.utils.create_shard`` without the
    Ray / S3 / thread-pool machinery.
    """
    shard_stem = f"shard-{shard_idx:04d}"
    shard_path = os.path.join(out_dir, f"{shard_stem}.tar")
    with tarfile.open(shard_path, "w") as tar:
        for sample in samples:
            for name, payload in sample.items():
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    return {"shard": shard_stem, "num_sequences": len(samples)}


def write_manifest(manifest_path: str, entries: list[dict[str, Any]]) -> None:
    """Write a ``manifest.jsonl`` file from a list of dicts.

    Mirrors ``vla_foundry.data.preprocessing.utils.save_and_upload_dict``
    for the local-filesystem / jsonl case.
    """
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def decode_video_frame_range(
    video_path: str | Path,
    start: int,
    count: int,
    *,
    fmt: str = "JPEG",
    quality: int = 95,
) -> list[bytes]:
    """Decode *count* frames starting at *start* from a video file.

    Returns a list of image bytes (encoded as *fmt*).

    Mirrors ``vla_foundry.data.preprocessing.robotics.converters.lerobot.decode_video_frames``
    but returns encoded bytes for a specific range instead of full-video numpy arrays.
    """
    frames: list[bytes] = []
    with av.open(str(video_path)) as container:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx < start:
                continue
            if idx >= start + count:
                break
            buf = io.BytesIO()
            frame.to_image().save(buf, format=fmt, quality=quality)
            frames.append(buf.getvalue())
    if len(frames) != count:
        raise ValueError(f"Expected {count} frames from {video_path}, got {len(frames)}.")
    return frames


def hf_download_with_retries(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = "dataset",
    attempts: int = 3,
    delay: float = 2.0,
) -> str:
    """Download a file from HuggingFace Hub with simple retry logic.

    Mirrors ``vla_foundry.data.preprocessing.hf_utils.hf_dataset_downloader._download_and_upload_file``
    but uses ``huggingface_hub.hf_hub_download`` (cached) and skips the
    S3-upload / exponential-backoff machinery.
    """
    for attempt in range(1, attempts + 1):
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
        except Exception:
            if attempt == attempts:
                raise
            print(f"Retrying {filename} ({attempt}/{attempts})...")
            time.sleep(delay)
    raise RuntimeError("unreachable")  # keeps mypy happy


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_training_loss(exp_dir: str, title: str | None = None) -> None:
    """Parse out.log from an experiment directory and plot loss + learning rate.

    Extracts per-step losses from the periodic log lines
    (``Train Checkpoint: ... Loss: X.XXX ... LR: X.XXXXXX``) to produce a
    dense, wandb-style training curve.
    """
    log_path = f"{exp_dir}/out.log"
    # Matches: Train Checkpoint: <ckpt> [<samples>/<total> (<pct>%)] Loss: <loss> ... LR: <lr>
    log_pattern = re.compile(r"Train Checkpoint: (\d+) \[\s*(\d+)/\d+.*?Loss: ([\d.]+).*?LR: ([\d.eE+-]+)")

    steps, losses, lrs = [], [], []
    ckpt_offsets: dict[int, int] = {}  # checkpoint_id -> cumulative sample offset
    cumulative = 0

    with open(log_path) as f:
        for line in f:
            m = log_pattern.search(line)
            if not m:
                continue
            ckpt_id = int(m.group(1))
            ckpt_samples = int(m.group(2))
            loss = float(m.group(3))
            lr = float(m.group(4))

            if ckpt_id not in ckpt_offsets:
                ckpt_offsets[ckpt_id] = cumulative

            global_samples = ckpt_offsets[ckpt_id] + ckpt_samples
            cumulative = max(cumulative, global_samples)

            steps.append(global_samples)
            losses.append(loss)
            lrs.append(lr)

    if not losses:
        print(f"No training logs found in {log_path}")
        return

    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.plot(steps, losses, "-", color="tab:blue", linewidth=1.2, alpha=0.85, label="Loss")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(steps, lrs, "--", color="tab:orange", linewidth=1, alpha=0.6, label="LR")
    ax2.set_ylabel("Learning Rate", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle(title or f"Training: {exp_dir.split('/')[-1]}", fontsize=11)
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.88), fontsize=8)
    plt.tight_layout()
    plt.show()
    print(f"Final loss: {losses[-1]:.4f} | LR: {lrs[-1]:.6f} | Samples: {steps[-1]}")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def copy_foundry_whitepaper_results(
    output_dir: str,
    models: list[str] | None = None,
    eval_set: str = "OSS",
    paper_results_dir: str = "vla_foundry/eval/eval_results",
) -> None:
    """Copy paper evaluation results into an output directory for dashboard comparison.

    Args:
        output_dir: Destination directory (e.g. ``tutorials/rollouts``).
        models: Specific model names to copy (e.g. ``["Qwen3VL2B-MT", "1.5B-VLA-FT"]``).
            If None, copies all models in the eval set.
        eval_set: Which result set to copy (``"OSS"`` or ``"CS"``).
        paper_results_dir: Path to the eval_results directory in the repo.
    """
    src = Path(paper_results_dir) / eval_set
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Copy rename.yaml (display names for dashboard).
    rename_src = Path(paper_results_dir) / "rename.yaml"
    if rename_src.exists() and not (dest / "rename.yaml").exists():
        shutil.copy2(rename_src, dest / "rename.yaml")
        print("Copied rename.yaml")

    available = sorted(d.name for d in src.iterdir() if d.is_dir())
    if models is None:
        to_copy = available
    else:
        missing = [m for m in models if m not in available]
        if missing:
            raise ValueError(f"Models not found in {src}: {missing}. Available: {available}")
        to_copy = models

    for name in to_copy:
        model_src = src / name
        model_dest = dest / name
        if not model_dest.exists():
            shutil.copytree(model_src, model_dest)
            print(f"Copied {name}")
        else:
            print(f"Skipped {name} (already exists)")


_TIMESTAMP_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T")


def clean_eval_timestamps(output_dir: str) -> None:
    """Remove loose timestamp directories at the root of an eval ``output_dir``.

    ``vla_foundry.eval.data_loading.load_episodes`` only accepts result files
    that live under ``{model}/{Task}/rollouts/{timestamp}/``. A timestamp dir
    sitting directly at ``output_dir`` root (e.g. from an older tutorial run
    with a different layout) trips its strict path check and raises
    ``ValueError``. Call this before re-running automated evaluation to keep
    the root clean while preserving ``{model}/...`` subdirectories.
    """
    root = Path(output_dir)
    if not root.is_dir():
        return
    for entry in root.iterdir():
        if entry.is_dir() and _TIMESTAMP_DIR_RE.match(entry.name):
            print(f"Removing stale timestamp dir: {entry}")
            shutil.rmtree(entry)


def download_lbm_spartan_raw(
    tasks: list[str],
    local_path: str = "tutorials/data/lbm_raw",
    force: bool = False,
) -> list[Path]:
    """Download raw LBM Spartan task tars and extract them under ``local_path``.

    Wraps ``vla_foundry/data/scripts/download_dataset.py --raw``. The script is
    itself idempotent (skips tar download on size-match and extraction when the
    task directory already exists), so this helper only adds a fast-path that
    avoids spawning a subprocess when extraction is already on disk.

    Args:
        tasks: Spartan task names to fetch, e.g. ``["PickAndPlaceBox"]``.
        local_path: Destination directory; tars land at ``<local_path>/<task>.tar``
            and extract to ``<local_path>/tasks/<task>/``.
        force: If True, re-download and re-extract each task even if a local
            copy already exists (passed through to ``--force``).

    Returns:
        Paths to each extracted task directory (``<local_path>/tasks/<task>``).
    """
    root = Path(local_path)
    extracted = []
    for task in tasks:
        task_dir = root / "tasks" / task
        if not force and task_dir.is_dir() and any(task_dir.iterdir()):
            print(f"Skipping {task}: already extracted at {task_dir}")
            extracted.append(task_dir)
            continue
        cmd = [
            sys.executable,
            "vla_foundry/data/scripts/download_dataset.py",
            "--task",
            task,
            "--local_path",
            str(root),
            "--raw",
        ]
        if force:
            cmd.append("--force")
        subprocess.run(cmd, check=True)
        extracted.append(task_dir)
    return extracted


# ---------------------------------------------------------------------------
# Dataset download functions
# ---------------------------------------------------------------------------


def download_dclm_text(
    out_dir: str = "tutorials/data/dclm_minimal",
    num_docs: int = 100_000,
    samples_per_shard: int = 4096,
) -> str:
    """Download DCLM-baseline text data and convert to WebDataset tars.

    Returns the path to the manifest file.
    """
    manifest_path = f"{out_dir}/manifest.jsonl"

    if os.path.exists(manifest_path):
        print(f"DCLM data already exists at {manifest_path}, skipping download.")
        return manifest_path

    print("Downloading one DCLM-baseline shard from Hugging Face...")
    path = hf_hub_download(
        repo_id="mlfoundations/dclm-baseline-1.0",
        filename="global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst",
        repo_type="dataset",
    )

    print(f"Reading the first {num_docs} documents...")
    docs = []
    with open(path, "rb") as f:
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        text_reader = io.TextIOWrapper(reader, encoding="utf-8")
        for i, line in enumerate(text_reader):
            if i >= num_docs:
                break
            docs.append(json.loads(line)["text"])

    os.makedirs(out_dir, exist_ok=True)
    manifest_entries = []
    for shard_idx, start in enumerate(range(0, len(docs), samples_per_shard)):
        batch = docs[start : start + samples_per_shard]
        samples = [
            {f"{start + j:08d}.json": json.dumps({"text": text}).encode("utf-8")} for j, text in enumerate(batch)
        ]
        manifest_entries.append(write_tar_shard(out_dir, shard_idx, samples))

    write_manifest(manifest_path, manifest_entries)
    print(f"Created {len(manifest_entries)} shards ({len(docs)} documents) in {out_dir}/")
    return manifest_path


def download_datacomp_images(
    out_dir: str = "tutorials/data/datacomp_minimal",
    num_samples: int = 64,
    samples_per_shard: int = 32,
) -> str:
    """Download DataComp-12M image-caption pairs and create WebDataset tars.

    Returns the path to the manifest file.
    """
    source_shard_filename = "00000000.tar"
    manifest_path = f"{out_dir}/manifest.jsonl"

    if os.path.exists(manifest_path):
        print(f"DataComp data already exists at {manifest_path}, skipping download.")
        return manifest_path

    print("Downloading one DataComp-12M shard index from Hugging Face...")
    cached_path = hf_hub_download(
        repo_id="mlfoundations/DataComp-12M",
        filename=source_shard_filename,
        repo_type="dataset",
    )

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    warnings.filterwarnings(
        "ignore",
        message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
        category=UserWarning,
    )

    examples = []
    timeout = 10
    session = requests.Session()

    with tarfile.open(cached_path, "r") as source_tar:
        grouped = {}
        for member in source_tar:
            if not member.isfile():
                continue
            if member.name.endswith(".url.txt"):
                stem = member.name[: -len(".url.txt")]
                suffix = "url.txt"
            else:
                stem, suffix = member.name.rsplit(".", 1)
            grouped.setdefault(stem, {})[suffix] = member

        print(f"Indexed {len(grouped)} candidate caption/url pairs from {source_shard_filename}.")
        attempted = 0
        for _key, files in grouped.items():
            if len(examples) >= num_samples:
                break
            if "txt" not in files or "url.txt" not in files:
                continue

            caption = source_tar.extractfile(files["txt"]).read().decode("utf-8").strip()
            image_url = source_tar.extractfile(files["url.txt"]).read().decode("utf-8").strip()
            if not caption or not image_url:
                continue

            attempted += 1
            try:
                response = session.get(image_url, timeout=timeout)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
            except Exception:
                if attempted % 25 == 0:
                    print(f"Attempted {attempted} URLs, kept {len(examples)} usable images so far...")
                continue

            image_bytes = io.BytesIO()
            image.save(image_bytes, format="JPEG", quality=90)
            examples.append(
                {
                    "key": f"{len(examples):08d}",
                    "caption": caption,
                    "url": image_url,
                    "image_bytes": image_bytes.getvalue(),
                }
            )
            if len(examples) % 8 == 0:
                print(f"Downloaded {len(examples)}/{num_samples} usable images after {attempted} URL attempts...")

    if len(examples) < samples_per_shard:
        raise RuntimeError(
            f"Only downloaded {len(examples)} usable images from DataComp; need at least {samples_per_shard}."
        )

    manifest_entries = []
    for shard_idx, start in enumerate(range(0, len(examples), samples_per_shard)):
        batch = examples[start : start + samples_per_shard]
        samples = [
            {
                f"{ex['key']}.txt": ex["caption"].encode("utf-8"),
                f"{ex['key']}.jpg": ex["image_bytes"],
                f"{ex['key']}.json": json.dumps({"url": ex["url"], "caption": ex["caption"]}).encode("utf-8"),
            }
            for ex in batch
        ]
        manifest_entries.append(write_tar_shard(out_dir, shard_idx, samples))

    write_manifest(manifest_path, manifest_entries)
    print(f"Built {len(manifest_entries)} local shards with {len(examples)} image-caption pairs in {out_dir}/")
    return manifest_path


def download_pixelprose_images(
    out_dir: str = "tutorials/data/pixelprose_minimal",
    num_samples: int = 64,
    samples_per_shard: int = 32,
) -> str:
    """Download PixelProse image-caption pairs and create WebDataset tars.

    Fetches a small parquet shard from tomg-group-umd/pixelprose (cc12m split),
    downloads images from their URLs, and packages them into local WebDataset tars.
    Uses the ``vlm_caption`` field (dense Gemini-generated captions).

    Returns the path to the manifest file.
    """
    manifest_path = f"{out_dir}/manifest.jsonl"

    if os.path.exists(manifest_path):
        print(f"PixelProse data already exists at {manifest_path}, skipping download.")
        return manifest_path

    print("Downloading one PixelProse parquet shard from Hugging Face...")
    parquet_path = hf_hub_download(
        repo_id="tomg-group-umd/pixelprose",
        filename="data/vlm_captions_cc12m_00.parquet",
        repo_type="dataset",
    )

    df = pd.read_parquet(parquet_path, columns=["url", "vlm_caption", "status"])
    # Keep only successfully captioned rows with valid captions
    df = df[(df["status"] == "success") & df["vlm_caption"].notna() & (df["vlm_caption"] != "")]
    df = df.reset_index(drop=True)
    print(f"Parquet shard has {len(df)} usable rows with vlm_caption.")

    os.makedirs(out_dir, exist_ok=True)

    warnings.filterwarnings(
        "ignore",
        message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
        category=UserWarning,
    )

    examples = []
    timeout = 10
    session = requests.Session()
    attempted = 0

    for _, row in df.iterrows():
        if len(examples) >= num_samples:
            break
        url = row["url"]
        caption = row["vlm_caption"].strip()
        if not url or not caption:
            continue

        attempted += 1
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception:
            if attempted % 25 == 0:
                print(f"Attempted {attempted} URLs, kept {len(examples)} usable images so far...")
            continue

        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG", quality=90)
        examples.append(
            {
                "key": f"{len(examples):08d}",
                "caption": caption,
                "url": url,
                "image_bytes": image_bytes.getvalue(),
            }
        )
        if len(examples) % 8 == 0:
            print(f"Downloaded {len(examples)}/{num_samples} usable images after {attempted} URL attempts...")

    if len(examples) < samples_per_shard:
        raise RuntimeError(
            f"Only downloaded {len(examples)} usable images from PixelProse; need at least {samples_per_shard}. "
            "Some URLs may be broken — try re-running."
        )

    manifest_entries = []
    for shard_idx, start in enumerate(range(0, len(examples), samples_per_shard)):
        batch = examples[start : start + samples_per_shard]
        samples = [
            {
                f"{ex['key']}.txt": ex["caption"].encode("utf-8"),
                f"{ex['key']}.jpg": ex["image_bytes"],
                f"{ex['key']}.json": json.dumps({"url": ex["url"], "caption": ex["caption"]}).encode("utf-8"),
            }
            for ex in batch
        ]
        manifest_entries.append(write_tar_shard(out_dir, shard_idx, samples))

    write_manifest(manifest_path, manifest_entries)
    print(f"Built {len(manifest_entries)} local shards with {len(examples)} image-caption pairs in {out_dir}/")
    return manifest_path


def download_droid_robotics(
    data_root: str = "tutorials/data/droid_100_minimal",
    num_episodes: int = 2,
) -> str:
    """Download DROID robotics data, extract frames, build LeRobot-compatible format.

    Returns the path to the lerobot-compatible data root (e.g. ``tutorials/data/droid_100_minimal/lerobot_compat``).
    """
    droid_cameras = [
        "observation.images.exterior_image_1_left",
        "observation.images.exterior_image_2_left",
        "observation.images.wrist_image_left",
    ]

    data_root = Path(data_root)
    compat_root = data_root / "lerobot_compat"
    compat_data_dir = compat_root / "data/chunk-000"

    if compat_data_dir.exists() and list(compat_data_dir.glob("*.parquet")):
        print(f"Using existing DROID compat data at {compat_root}.")
        return str(compat_root)

    def slice_episode_rows(df, episode_row):
        start = int(episode_row["dataset_from_index"])
        stop = int(episode_row["dataset_to_index"])
        expected = int(episode_row["length"])
        rows = df.iloc[start:stop].copy()
        if len(rows) != expected:
            rows = df.iloc[start : stop + 1].copy()
        if len(rows) != expected:
            raise ValueError(f"Could not match episode length {expected} for slice [{start}, {stop}].")
        return start, rows

    repo = "lerobot/droid_100"
    print("Downloading DROID metadata and camera videos...")
    info_path = hf_download_with_retries(repo, "meta/info.json")
    episodes_path = hf_download_with_retries(repo, "meta/episodes/chunk-000/file-000.parquet")
    hf_download_with_retries(repo, "meta/tasks.parquet")
    data_path = hf_download_with_retries(repo, "data/chunk-000/file-000.parquet")
    video_paths = {
        camera: Path(hf_download_with_retries(repo, f"videos/{camera}/chunk-000/file-000.mp4"))
        for camera in droid_cameras
    }

    if data_root.exists():
        shutil.rmtree(data_root)
    (compat_root / "meta").mkdir(parents=True, exist_ok=True)
    compat_data_dir.mkdir(parents=True, exist_ok=True)

    with open(info_path) as f:
        info = json.load(f)
    episodes_df = pq.read_table(episodes_path).to_pandas()
    data_df = pq.read_table(data_path).to_pandas()

    selected = episodes_df.head(num_episodes).copy()
    task_entries = []
    episode_entries = []

    for local_idx, (_, ep_row) in enumerate(selected.iterrows()):
        global_start, ep_rows = slice_episode_rows(data_df, ep_row)
        task_text = str(ep_row["tasks"][0])
        print(
            f"Building local episode {local_idx} "
            f"from source episode {int(ep_row['episode_index'])} "
            f"with {len(ep_rows)} frames..."
        )

        frame_count = len(ep_rows)
        for camera in droid_cameras:
            ep_rows[camera] = decode_video_frame_range(video_paths[camera], global_start, frame_count)

        ep_rows = ep_rows.copy()
        ep_rows["episode_index"] = local_idx
        ep_rows["frame_index"] = list(range(frame_count))
        ep_rows["index"] = list(range(frame_count))
        task_index = len(task_entries)
        ep_rows["task_index"] = task_index

        out_path = compat_data_dir / f"episode_{local_idx:06d}.parquet"
        pq.write_table(pa.Table.from_pandas(ep_rows, preserve_index=False), out_path)

        task_entries.append({"task_index": task_index, "task": task_text})
        episode_entries.append({"episode_index": local_idx, "tasks": [task_text]})

    with open(compat_root / "meta/info.json", "w") as f:
        json.dump({"fps": info["fps"]}, f)
    write_manifest(str(compat_root / "meta/episodes.jsonl"), episode_entries)
    write_manifest(str(compat_root / "meta/tasks.jsonl"), task_entries)

    print(f"Built LeRobot-compatible data at {compat_root}/")
    return str(compat_root)
