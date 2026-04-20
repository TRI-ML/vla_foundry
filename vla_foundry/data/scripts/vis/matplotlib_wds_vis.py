#!/usr/bin/env python3
"""
WDS TAR Viewer — stream contents from S3 and visualize.

Features
- Stream one or many .tar shard(s) directly from S3 (no local download required)
- For low-dimensional npz payloads: print keys and array shapes
- For images (jpg/png): render a contact sheet (optionally save to disk)

Examples
Single shard (stream) and view everything interactively:
    python matplotlib_wds_vis.py s3://my-bucket/path/shard-00001.tar --show

Multiple shards under a prefix (save contact sheets, don't pop windows):
    python matplotlib_wds_vis.py s3://my-bucket/path/ --recursive --save --output-dir ./reports

Only print lowdim shapes:
    python matplotlib_wds_vis.py s3://my-bucket/path/shard-00001.tar --no-images

Limit images (set -1 to show all):
    python matplotlib_wds_vis.py s3://my-bucket/path/shard-00001.tar --max-images 200


Notes
- Uses tarfile streaming mode ("r|*") for constant memory iteration.
- Contact sheets are saved as PNG files; one per tar.
"""

import argparse
import io
import os
import sys
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass

import fsspec

# Matplotlib is imported lazily so headless servers without a DISPLAY still work when --save only
import matplotlib
import numpy as np
from PIL import Image
from tqdm import tqdm

SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png"}
LOWDIM_EXTS = {".npz"}


@dataclass
class LowdimSummary:
    member_name: str
    keys: list[str]
    shapes: dict[str, tuple[int, ...]]


@dataclass
class TarReport:
    s3_path: str
    lowdim_summaries: list[LowdimSummary]
    image_entries: list[tuple[str, Image.Image]]  # (member_name, PIL Image)


def is_image(name: str) -> bool:
    n = name.lower()
    return any(n.endswith(ext) for ext in SUPPORTED_IMG_EXTS)


def is_lowdim(name: str) -> bool:
    n = name.lower()
    return any(n.endswith(ext) for ext in LOWDIM_EXTS) and ("lowdim" in n or "npz" in n)


def open_s3(path: str):
    if not path.startswith("s3://"):
        raise ValueError(f"Expect s3:// URI, got: {path}")
    fs, _ = fsspec.core.url_to_fs(path)
    return fs.open(path, "rb")


def iter_tar_members(fileobj: io.BufferedReader) -> Iterable[tarfile.TarInfo]:
    # Streaming mode: sequential read, minimal memory
    with tarfile.open(fileobj=fileobj, mode="r|*") as tf:
        yield from tf


def extract_member_bytes(fileobj: io.BufferedReader, member: tarfile.TarInfo) -> bytes:
    # With streaming tarfile, after TarInfo is yielded, the next .extractfile() reads its content now
    with tarfile.open(fileobj=fileobj, mode="r|*") as tf:
        # We need to iterate until we reach the requested member in order again.
        for m in tf:
            if m.name == member.name:
                f = tf.extractfile(m)
                if f is None:
                    return b""
                return f.read()
    return b""


def scan_tar_stream(s3_path: str, max_images: int = -1) -> TarReport:
    lowdim_summaries: list[LowdimSummary] = []
    image_entries: list[tuple[str, Image.Image]] = []

    # Open one streaming reader we will iterate exactly once; because streaming tar
    # cannot seek backwards, we must read members and their bytes on the fly.
    with open_s3(s3_path) as fo, tarfile.open(fileobj=fo, mode="r|*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name
            # Load content now; after this iteration we cannot come back.
            f = tf.extractfile(m)
            if f is None:
                continue
            data = f.read()

            if is_lowdim(name):
                try:
                    with np.load(io.BytesIO(data), allow_pickle=True) as npz:
                        keys = list(npz.keys())
                        shapes = {k: tuple(np.array(npz[k]).shape) for k in keys}
                    lowdim_summaries.append(LowdimSummary(name, keys, shapes))
                except Exception:
                    lowdim_summaries.append(LowdimSummary(name + " (failed)", ["error"], {"error": tuple()}))
            elif is_image(name):
                if max_images == -1 or len(image_entries) < max_images:
                    try:
                        im = Image.open(io.BytesIO(data)).convert("RGB")
                        image_entries.append((name, im))
                    except Exception:
                        pass
            else:
                # ignore other payloads
                pass

    return TarReport(s3_path=s3_path, lowdim_summaries=lowdim_summaries, image_entries=image_entries)


def ensure_matplotlib_backend(show: bool):
    # If not showing, we can safely use Agg
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401


def render_contact_sheet(images: list[tuple[str, Image.Image]], title: str, save_path: str | None, show: bool):
    ensure_matplotlib_backend(show)
    import matplotlib.pyplot as plt

    if not images:
        return

    n = len(images)
    # grid heuristic: ~square
    cols = int(max(1, round(n**0.5)))
    rows = (n + cols - 1) // cols

    fig_w = min(16, max(6, cols * 3))
    fig_h = min(16, max(6, rows * 3))
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title)

    for i, (name, im) in enumerate(images, start=1):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(im)
        ax.set_title(os.path.basename(name), fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def print_lowdim_summary(report: TarReport):
    if not report.lowdim_summaries:
        print(f"[lowdim] No npz found in {report.s3_path}")
        return
    print(f"[lowdim] {report.s3_path}")
    for s in report.lowdim_summaries:
        print(f"  - {s.member_name}")
        for k in s.keys:
            shape = s.shapes.get(k, ())
            print(f"      {k}: {shape}")


def list_s3_targets(uri: str, recursive: bool) -> list[str]:
    if uri.endswith(".tar"):
        return [uri]
    # Treat as prefix
    if not uri.endswith("/"):
        uri = uri + "/"
    fs, _ = fsspec.core.url_to_fs(uri)
    # Use glob to find tars under prefix
    pat = uri + ("**/*.tar" if recursive else "*.tar")
    paths = sorted(fs.glob(pat))
    # fs.glob returns paths without s3:// sometimes; normalize
    norm = []
    for p in paths:
        if not str(p).startswith("s3://"):
            norm.append("s3://" + str(p))
        else:
            norm.append(str(p))
    return norm


def parse_args():
    p = argparse.ArgumentParser(description="Stream WDS tar(s) from S3 and visualize contents")
    p.add_argument("s3_uri", help="s3://bucket/prefix or s3://bucket/key.tar")
    p.add_argument("--recursive", action="store_true", help="recurse under prefix to find .tar shards")
    p.add_argument("--max-images", type=int, default=100, help="limit images per tar (-1 = all)")
    p.add_argument("--no-images", action="store_true", help="skip image rendering")
    p.add_argument("--show", action="store_true", help="display contact sheets in a window")
    p.add_argument("--save", action="store_true", help="save contact sheets to --output-dir")
    p.add_argument("--output-dir", default="./wds_reports", help="directory to save outputs when --save is used")
    return p.parse_args()


def main():
    args = parse_args()

    targets = list_s3_targets(args.s3_uri, args.recursive)
    if not targets:
        print("No .tar files found for given URI and options.")
        sys.exit(2)

    if not args.show and not args.save and len(targets) == 1:
        # default to showing interactively for a single target, otherwise only print
        args.show = True

    for s3_path in tqdm(targets, desc="Tars"):
        try:
            report = scan_tar_stream(s3_path, max_images=args.max_images if not args.no_images else 0)
        except Exception as e:
            print(f"Error reading {s3_path}: {e}")
            continue

        print_lowdim_summary(report)

        if not args.no_images and report.image_entries:
            save_path = None
            if args.save:
                # sanitize filename from s3 key
                key = s3_path.replace("s3://", "").replace("/", "_")
                save_path = os.path.join(args.output_dir, f"contactsheet_{key}.png")
            title = f"{s3_path} — {len(report.image_entries)} images"
            render_contact_sheet(report.image_entries, title, save_path, show=args.show)
        elif not args.no_images:
            print(f"[images] No images found in {s3_path}")


if __name__ == "__main__":
    main()
