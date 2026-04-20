#!/usr/bin/env python3
"""
Webdataset TAR Viewer — Gradio edition (stream from S3, view lowdim + images)

Features
- Point at an S3 prefix or single shard and scan for .tar files
- Select a tar and stream its contents (no full download)
- Show lowdim .npz keys & shapes in a table
- Show all images (jpg/png) as a gallery

Usage
    python gradio_wds_vis.py

Notes
- Uses tarfile streaming mode ("r|*") for constant-memory iteration
- Gallery can be limited by a slider; set to -1 to load all images
"""

import io
import os
import tarfile
from dataclasses import dataclass

import fsspec
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png"}
LOWDIM_EXTS = {".npz"}


@dataclass
class LowdimSummary:
    member_name: str
    keys: list[str]
    shapes: dict[str, tuple[int, ...]]


def is_image(name: str) -> bool:
    n = name.lower()
    return any(n.endswith(ext) for ext in SUPPORTED_IMG_EXTS)


def is_lowdim(name: str) -> bool:
    n = name.lower()
    return any(n.endswith(ext) for ext in LOWDIM_EXTS) and ("lowdim" in n or "npz" in n)


def list_s3_targets(uri: str, recursive: bool) -> list[str]:
    if not uri:
        return []
    if uri.endswith(".tar"):
        return [uri]
    # Treat as prefix
    if not uri.endswith("/"):
        uri = uri + "/"
    fs, _ = fsspec.core.url_to_fs(uri)
    pattern = uri + ("**/*.tar" if recursive else "*.tar")
    paths = sorted(fs.glob(pattern))
    # Normalize to s3://
    norm = []
    for p in paths:
        p = str(p)
        if not p.startswith("s3://"):
            p = "s3://" + p
        norm.append(p)
    return norm


def scan_tar_stream(s3_path: str, max_images: int = -1):
    """Return (lowdim_df, gallery_items) for a single .tar streamed from S3.
    gallery_items is list of (PIL.Image, caption)
    """
    lowdim_rows: list[dict[str, str]] = []
    gallery_items: list[tuple[Image.Image, str]] = []

    if not s3_path:
        return pd.DataFrame(), []

    fs, _ = fsspec.core.url_to_fs(s3_path)
    with fs.open(s3_path, "rb") as fo, tarfile.open(fileobj=fo, mode="r|*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name
            ef = tf.extractfile(m)
            if ef is None:
                continue
            data = ef.read()

            if is_lowdim(name):
                try:
                    with np.load(io.BytesIO(data), allow_pickle=True) as npz:
                        for k in npz:
                            arr = np.array(npz[k])
                            lowdim_rows.append(
                                {
                                    "member": name,
                                    "key": k,
                                    "shape": str(tuple(arr.shape)),
                                    "dtype": str(arr.dtype),
                                }
                            )
                except Exception as e:
                    lowdim_rows.append(
                        {
                            "member": name,
                            "key": "<error>",
                            "shape": "-",
                            "dtype": f"{type(e).__name__}",
                        }
                    )
            elif is_image(name):
                if max_images == -1 or len(gallery_items) < max_images:
                    try:
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        # optional: thumbnail to keep UI snappy
                        img.thumbnail((1024, 1024))
                        gallery_items.append((img, os.path.basename(name)))
                    except Exception:
                        pass
            else:
                # ignore other members
                pass

    df = pd.DataFrame(lowdim_rows)
    return df, gallery_items


# =====================
# Gradio UI
# =====================


def ui_scan(prefix: str, recursive: bool):
    tars = list_s3_targets(prefix.strip(), recursive)
    msg = f"Found {len(tars)} tar(s)." if tars else "No .tar files found."
    return gr.update(choices=tars, value=(tars[0] if tars else None)), msg


def ui_load(s3_tar: str, max_images: int):
    if not s3_tar:
        return pd.DataFrame(), [], "Select a tar first."
    df, gallery = scan_tar_stream(s3_tar, max_images=max_images)
    note = f"Loaded {len(gallery)} image(s)." if gallery else "No images in this tar."
    return df, gallery, note


def build_app():
    with gr.Blocks(title="WDS TAR Viewer — Gradio") as demo:
        gr.Markdown("""
        # WDS TAR Viewer — Gradio
        Stream **WebDataset** .tar shards from **S3**, view **lowdim** shapes and **images**.
        """)

        with gr.Row():
            s3_prefix = gr.Textbox(
                label="S3 URI (prefix or .tar)", placeholder="s3://bucket/path or s3://bucket/key.tar"
            )
            recursive = gr.Checkbox(label="Recursive", value=True)
            scan_btn = gr.Button("Scan")
        scan_status = gr.Markdown(visible=True)

        with gr.Row():
            tar_select = gr.Dropdown(label="Select TAR shard", choices=[], interactive=True)
            max_images = gr.Slider(visible=True, label="Max images", minimum=-1, maximum=1000, step=1, value=200)
            load_btn = gr.Button("Load TAR")

        with gr.Tab("Lowdim"):
            lowdim_table = gr.Dataframe(
                headers=["member", "key", "shape", "dtype"], row_count=(1, "dynamic"), wrap=True
            )
        with gr.Tab("Images"):
            gallery = gr.Gallery(label="Images", columns=4, height=700, preview=True)
        note = gr.Markdown()

        # Wire
        scan_btn.click(ui_scan, inputs=[s3_prefix, recursive], outputs=[tar_select, scan_status])
        # convenience: if user passes a direct .tar, pressing scan will set dropdown; pressing Load then fetches
        load_btn.click(ui_load, inputs=[tar_select, max_images], outputs=[lowdim_table, gallery, note])

        gr.Markdown("""
        **Tips**
        - If you paste a direct `s3://...tar` above, hit **Scan** to populate the selector, then **Load TAR**.
        - Set **Max images** to `-1` to load all images.
        - Only `.npz` lowdim members are parsed for key/shape.
        """)
    return demo


if __name__ == "__main__":
    app = build_app()
    # queue enables responsive UI while streaming results
    app.queue().launch()
