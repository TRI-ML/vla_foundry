"""VLM inference script. Generates captions for images.

Usage:
    # Run on default test images
    python -m vla_foundry.inference.inference_vlm <experiment_dir>

    # Run on specific images
    python -m vla_foundry.inference.inference_vlm <experiment_dir> img1.jpg img2.jpg

    # Specify device
    python -m vla_foundry.inference.inference_vlm <experiment_dir> --device cuda:1

    experiment_dir: directory containing config.yaml and checkpoints/
"""

import argparse
import glob
import os

import requests
import torch
from PIL import Image, ImageOps

from vla_foundry.data.processor import apply_chat_template, get_processor
from vla_foundry.file_utils import load_model_checkpoint
from vla_foundry.models import create_model
from vla_foundry.params.train_experiment_params import TrainExperimentParams, load_params_from_yaml

DEFAULT_TEST_IMAGES = [
    ("cat", "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"),
    ("car", "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"),
]


def load_image(path, center_crop_square=False, pad_square=False):
    if path.startswith("http"):
        img = Image.open(requests.get(path, stream=True).raw).convert("RGB")
    else:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert("RGB")
    if pad_square:
        w, h = img.size
        s = max(w, h)
        padded = Image.new("RGB", (s, s), (255, 255, 255))
        padded.paste(img, ((s - w) // 2, (s - h) // 2))
        img = padded
    elif center_crop_square:
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
    return img


def main():
    parser = argparse.ArgumentParser(description="VLM inference / vibe check")
    parser.add_argument("experiment_dir", help="Directory with config.yaml and checkpoints/")
    parser.add_argument("images", nargs="*", help="Image paths or URLs (default: test images)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint filename (default: latest)")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--center-crop", action="store_true", help="Center crop images to square before processing")
    parser.add_argument("--pad-square", action="store_true", help="Pad images symmetrically with white to make square")
    parser.add_argument("--save-dir", default=None, help="Directory to save processed images")
    args = parser.parse_args()

    # Load config
    train_params = load_params_from_yaml(TrainExperimentParams, f"{args.experiment_dir}/config.yaml")

    # Create model (load_pretrained=False skips S3 checkpoint downloads —
    # we load the full checkpoint separately)
    model = create_model(train_params.model, load_pretrained=False)
    if args.checkpoint:
        ckpt_path = f"{args.experiment_dir}/checkpoints/{args.checkpoint}"
    else:
        ckpts = sorted(glob.glob(f"{args.experiment_dir}/checkpoints/checkpoint_*.pt"))
        ckpt_path = ckpts[-1] if ckpts else f"{args.experiment_dir}/checkpoints/checkpoint_1.pt"
    print(f"Loading {ckpt_path}")
    load_model_checkpoint(model, ckpt_path)

    precision = getattr(train_params.hparams, "precision", "pure_bf16")
    dtype = torch.bfloat16 if "bf16" in precision else torch.float16 if "fp16" in precision else torch.float32
    model = model.to(args.device).to(dtype).eval()

    # Load processor
    processor = get_processor(train_params.data)
    proc_kwargs = getattr(train_params.data, "processor_kwargs", {})
    eos_id = getattr(processor.tokenizer, "eos_token_id", None)
    prompt = apply_chat_template(processor, 1, "")

    # Build image list
    images = [(path, path) for path in args.images] if args.images else DEFAULT_TEST_IMAGES

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for name, path in images:
        image = load_image(path, center_crop_square=args.center_crop, pad_square=args.pad_square)
        if args.save_dir:
            basename = os.path.splitext(os.path.basename(name))[0] + ".png"
            image.save(os.path.join(args.save_dir, basename))
        inputs = processor(image, prompt, return_tensors="pt", **proc_kwargs)
        if inputs["pixel_values"].dim() == 5:
            inputs["pixel_values"] = inputs["pixel_values"].squeeze(1)
        inputs.pop("pixel_attention_mask", None)

        # Strip trailing <end_of_utterance> + \n to match training format
        if eos_id is not None and inputs["input_ids"].shape[-1] >= 2 and inputs["input_ids"][0, -2].item() == eos_id:
            inputs["input_ids"] = inputs["input_ids"][:, :-2]
            inputs["attention_mask"] = inputs["attention_mask"][:, :-2]

        inputs = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature, eos_token_id=eos_id
            )

        prompt_len = inputs["input_ids"].shape[-1]
        decoded = processor.decode(out[0][prompt_len:], skip_special_tokens=True)
        print(f"{name}: {decoded}")


if __name__ == "__main__":
    main()
