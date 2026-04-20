"""
python vla_foundry/data/scripts/vis/dataloader_vis.py (--arguments-here)

This uses the same argument parsing `draccus.parse(config_class=TrainExperimentParams)` as `main.py`
so you can take any script in `./examples` and copy paste the arguments exactly.

It will load the dataloader the exact same way that `main.py` loads it.
"""

### This part is copy-pasted from main.py

import logging
import os

import draccus
import gradio as gr
import numpy as np
import torch

from vla_foundry.data.dataloader import get_datastring_input, get_wds_dataloader
from vla_foundry.logger import setup_logging
from vla_foundry.params.train_experiment_params import TrainExperimentParams
from vla_foundry.utils import get_experiment_name, set_random_seed

cfg = draccus.parse(config_class=TrainExperimentParams)

### Force batch size to 1
object.__setattr__(cfg.hparams, "per_gpu_batch_size", 1)
object.__setattr__(cfg.hparams, "global_batch_size", 1)

device = cfg.distributed.device
# Seed rank-0 before any object creation for reproducibility.
set_random_seed(cfg.hparams.seed, 0)

# Set path for experiment, log, checkpoints.
experiment_name = get_experiment_name(cfg)
if cfg.save_path is None:
    experiment_path = os.path.join("experiments", experiment_name)
else:
    experiment_path = os.path.join(cfg.save_path, experiment_name)
os.makedirs(experiment_path, exist_ok=True)
log_path = os.path.join(experiment_path, "out.log")
setup_logging(log_path, logging.INFO)
checkpoint_path = os.path.join(experiment_path, "checkpoints")
os.makedirs(checkpoint_path, exist_ok=True)

start_checkpoint_num = 0
shard_shuffle_seed_per_dataset = None
checkpoint_num = start_checkpoint_num
# Per-dataset cursors and shuffle seeds allow resuming mixed datasets.
curr_shard_idx_per_dataset = [0 for dataset in range(len(cfg.data.dataset_manifest))]
if shard_shuffle_seed_per_dataset is None:
    shard_shuffle_seed_per_dataset = [cfg.hparams.seed for dataset in range(len(cfg.data.dataset_manifest))]

samples_seen = 0

# Partition the global sample budget into evenly-sized checkpoint chunks.
samples_per_checkpoint = cfg.total_train_samples // cfg.num_checkpoints
datastrings, num_samples_per_dataset, curr_shard_idx_per_dataset, shard_shuffle_seed_per_dataset = get_datastring_input(
    num_samples=samples_per_checkpoint,
    curr_shard_idx_per_dataset=curr_shard_idx_per_dataset,
    shard_shuffle_seed_per_dataset=shard_shuffle_seed_per_dataset,
    manifest_paths=cfg.data.dataset_manifest,
    dataset_weighting=cfg.data.dataset_weighting,
    allow_multiple_epochs=cfg.data.allow_multiple_epochs,
    num_workers_per_gpu=cfg.data.num_workers,
    world_size=cfg.distributed.world_size,
)

logging.info(f"Now training on: {datastrings}")
logging.info(f"Samples: {samples_seen} / {cfg.total_train_samples}")
logging.info(f"Samples in this checkpoint (per dataset): {num_samples_per_dataset}")

# Safety check: ensure all ranks see the same data slice.
if cfg.distributed.use_distributed:
    all_datastrings = ["" for _ in range(cfg.distributed.world_size)]
    torch.distributed.all_gather_object(all_datastrings, datastrings)
    assert all([x == datastrings for x in all_datastrings]), (
        "Dataset to train on is not the same across all nodes. This should not happen normally, "
        "unless there is an issue with shard shuffling during the dataset generation."
    )

dataloader = get_wds_dataloader(datastrings, num_samples_per_dataset, checkpoint_num, cfg)


### Visualization code starts here
num_samples_to_display = 10
samples = []
for idx, batch in enumerate(dataloader.dataloader):
    samples.append(batch)
    if idx >= num_samples_to_display:
        break

### Visualize samples
# Get all unique keys from all samples
all_keys = set()
for sample in samples:
    all_keys.update(sample.keys())
all_keys = sorted(list(all_keys))


def get_field_content(batch_idx, field_name):
    if batch_idx >= len(samples):
        return "Batch index out of range"

    batch = samples[batch_idx]
    if field_name not in batch:
        return f"Field '{field_name}' not found in batch {batch_idx}"

    value = batch[field_name]
    content = f"Field: {field_name}\n"

    if hasattr(value, "shape"):
        content += f"Shape: {value.shape}, Type: {type(value)}"
        if hasattr(value, "dtype"):
            content += f", Dtype: {value.dtype}"
        content += "\n\nContent:\n"

        # Show actual tensor values (but not for image fields)
        if torch.is_tensor(value):
            if field_name in ["pixel_values", "images"]:
                content += f"[Image tensor - see gallery below]\nMin: {value.min().item()}, Max: {value.max().item()}"
                if value.dtype.is_floating_point:
                    content += f", Mean: {value.mean().item():.4f}"
                else:
                    content += f", Mean: {value.float().mean().item():.4f}"
            else:
                content += str(value.tolist())
        else:
            content += str(value)

    elif isinstance(value, list):
        content += f"Type: list with {len(value)} items\n\nContent:\n"
        for i, item in enumerate(value):
            if hasattr(item, "keys"):  # Dict-like
                content += f"Item {i}: {dict(item)}\n"
            elif hasattr(item, "shape"):  # Tensor-like
                content += f"Item {i}: shape {item.shape}, type {type(item)}\n"
            else:
                content += f"Item {i}: {item}\n"

    elif isinstance(value, dict):
        content += f"Type: dict with {len(value)} keys\n\nContent:\n"
        for k, v in value.items():
            if hasattr(v, "shape"):
                content += f"{k}: shape {v.shape}, type {type(v)}\n"
            else:
                content += f"{k}: {v}\n"

    else:
        content += f"Type: {type(value)}\n\nContent:\n"
        content += str(value)

    return content


def get_field_images(batch_idx, field_name):
    if batch_idx >= len(samples):
        return None

    batch = samples[batch_idx]
    if field_name not in batch:
        return None

    value = batch[field_name]

    # Handle pixel_values
    if field_name == "pixel_values" and value is not None:
        try:
            pixel_values = value[0]  # First sample
            images = []

            # Check if this is a single RGB image (3, H, W) or multiple images (N, 3, H, W)
            if len(pixel_values.shape) == 3 and pixel_values.shape[0] == 3:
                # This is likely a single RGB image (3, H, W)
                try:
                    img_tensor = pixel_values

                    # Skip if tensor is too small
                    if img_tensor.numel() < 12:  # 3 channels * 2x2 minimum
                        return None

                    if img_tensor.shape[1] < 2 or img_tensor.shape[2] < 2:  # Skip tiny images
                        return None

                    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)

                    # Normalize to [0, 1]
                    if img.min() < 0:  # Denormalize if needed
                        img = (img + 1) / 2
                    img = np.clip(img, 0, 1)

                    # Convert to uint8
                    img_uint8 = (img * 255).astype(np.uint8)
                    images.append(img_uint8)

                except Exception as e:
                    print(f"Error processing single RGB image: {e}")
                    return None
            else:
                # Multiple images case
                for i in range(pixel_values.shape[0]):
                    try:
                        img_tensor = pixel_values[i]

                        # Skip if tensor is too small or has weird dimensions
                        if img_tensor.numel() < 4:  # Skip tiny tensors
                            continue

                        # Handle different tensor dimensions
                        if len(img_tensor.shape) == 3:  # (C, H, W)
                            if img_tensor.shape[1] < 2 or img_tensor.shape[2] < 2:  # Skip tiny images
                                continue
                            img = img_tensor.permute(1, 2, 0).cpu().numpy()
                        elif len(img_tensor.shape) == 2:  # (H, W) - grayscale
                            if img_tensor.shape[0] < 2 or img_tensor.shape[1] < 2:  # Skip tiny images
                                continue
                            img = img_tensor.cpu().numpy()
                            img = np.expand_dims(img, axis=2)  # Add channel dimension
                        else:
                            continue  # Skip unsupported dimensions

                        # Normalize to [0, 1]
                        if img.min() < 0:  # Denormalize if needed
                            img = (img + 1) / 2
                        img = np.clip(img, 0, 1)

                        # Convert to uint8 and ensure proper shape
                        img_uint8 = (img * 255).astype(np.uint8)

                        # Ensure we have at least 2x2 image
                        if img_uint8.shape[0] < 2 or img_uint8.shape[1] < 2:
                            continue

                        # Handle grayscale vs RGB
                        if img_uint8.shape[2] == 1:
                            img_uint8 = np.repeat(img_uint8, 3, axis=2)  # Convert grayscale to RGB
                        elif img_uint8.shape[2] > 3:
                            img_uint8 = img_uint8[:, :, :3]  # Take only first 3 channels

                        images.append(img_uint8)
                    except Exception as e:
                        print(f"Skipping image {i}: {e}")
                        continue

            return images if images else None
        except Exception as e:
            print(f"Error processing pixel_values: {e}")
            return None

    # Handle raw images
    if field_name == "images" and value is not None:
        try:
            if isinstance(value, list) and len(value) > 0:
                sample_images = value[0]  # First sample
                if isinstance(sample_images, dict):
                    images = []
                    for camera_name, img in sample_images.items():
                        try:
                            img_array = img if isinstance(img, np.ndarray) else np.array(img)

                            # Skip tiny or invalid images
                            if img_array.size < 4 or len(img_array.shape) < 2:
                                continue
                            if img_array.shape[0] < 2 or img_array.shape[1] < 2:
                                continue

                            # Ensure uint8 format
                            if img_array.dtype != np.uint8:
                                if img_array.max() <= 1.0:
                                    img_array = (img_array * 255).astype(np.uint8)
                                else:
                                    img_array = img_array.astype(np.uint8)

                            # Handle different channel configurations
                            if len(img_array.shape) == 2:  # Grayscale
                                img_array = np.stack([img_array] * 3, axis=2)
                            elif len(img_array.shape) == 3 and img_array.shape[2] == 1:  # Single channel
                                img_array = np.repeat(img_array, 3, axis=2)
                            elif len(img_array.shape) == 3 and img_array.shape[2] > 3:  # Too many channels
                                img_array = img_array[:, :, :3]

                            images.append(img_array)
                        except Exception as e:
                            print(f"Skipping image {camera_name}: {e}")
                            continue
                    return images if images else None
        except Exception as e:
            print(f"Error processing raw images: {e}")
            return None

    return None


# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Dataloader Visualization")

    with gr.Row():
        batch_slider = gr.Slider(0, len(samples) - 1, value=0, step=1, label="Batch Index")

    # Create a separate box for each field
    field_components = {}
    for field_name in all_keys:
        with gr.Row(), gr.Column():
            gr.Markdown(f"## {field_name}")
            field_info = gr.Textbox(label=f"{field_name} Info", lines=8)
            # Always show gallery for image-related fields
            show_gallery = field_name in ["pixel_values", "images"]
            field_images = gr.Gallery(label=f"{field_name} Images", columns=2, visible=show_gallery)

            field_components[field_name] = {"info": field_info, "images": field_images}

    # Set up event handlers
    def update_all_fields(batch_idx):
        outputs = []
        for field_name in all_keys:
            content = get_field_content(batch_idx, field_name)
            images = get_field_images(batch_idx, field_name)
            outputs.append(content)
            if images is not None:
                field_components[field_name]["images"].visible = True
                outputs.append(images)
            else:
                field_components[field_name]["images"].visible = False
                outputs.append(None)
        return outputs

    # Create output list for all components
    all_outputs = []
    for field_name in all_keys:
        all_outputs.append(field_components[field_name]["info"])
        all_outputs.append(field_components[field_name]["images"])

    batch_slider.change(fn=update_all_fields, inputs=[batch_slider], outputs=all_outputs)

demo.launch()
