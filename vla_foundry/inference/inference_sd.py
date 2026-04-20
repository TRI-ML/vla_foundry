import argparse
import os

import torch
from PIL import Image

from vla_foundry.file_utils import load_model_checkpoint
from vla_foundry.models import create_model
from vla_foundry.params.model_params import ModelParams
from vla_foundry.params.train_experiment_params import load_params_from_yaml

"""
This script generates images using a Stable Diffusion model.
Example usage:
    With prompt (text conditioning): uv run vla_foundry/inference/inference_sd.py --prompt="A picture of a cat"
    Without prompt: uv run vla_foundry/inference/inference_sd.py
"""

args = argparse.ArgumentParser()
args.add_argument("--prompt", type=str, default="tree")
prompt = args.parse_args().prompt


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


# Load config and model
model_params = load_params_from_yaml(
    ModelParams,
    "s3://your-bucket/your-path/stable_diffusion_cfg/config.yaml",
)
model = create_model(model_params, load_pretrained=False)
ckpt = "s3://your-bucket/your-path/stable_diffusion_cfg/checkpoints/checkpoint_12.pt"
load_model_checkpoint(model, ckpt)

model = model.to("cuda")
images = model.generate(batch_size=16, device=torch.device("cuda"), prompt=prompt)  # numpy [16, 224, 224, 3]

# Make a grid and save the images
image_grid = make_grid(images, rows=4, cols=4)
output_dir = "experiments/"
os.makedirs(output_dir, exist_ok=True)
image_grid.save(f"{output_dir}/stable_diffusion_output_{prompt}.png")
