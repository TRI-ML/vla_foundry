import sys

import torch
from transformers import AutoTokenizer

from vla_foundry.file_utils import load_model_checkpoint
from vla_foundry.models import create_model
from vla_foundry.params.train_experiment_params import TrainExperimentParams, load_params_from_yaml

if len(sys.argv) < 2:
    print("Usage: python inference.py <base_path> [checkpoint_name]")
    print("  base_path: directory containing config.yaml and checkpoints/")
    print("  checkpoint_name: (optional) checkpoint file, defaults to checkpoint_1.pt")
    sys.exit(1)

BASE_PATH = sys.argv[1]
CHECKPOINT_NAME = sys.argv[2] if len(sys.argv) > 2 else "checkpoint_1.pt"
CHECKPOINT = f"{BASE_PATH}/checkpoints/{CHECKPOINT_NAME}"

print(f"Loading config from {BASE_PATH}...")
train_params = load_params_from_yaml(TrainExperimentParams, f"{BASE_PATH}/config.yaml")

print("Creating model...")
model = create_model(train_params.model, load_pretrained=False)
model = model.cuda()

print(f"Loading checkpoint from {CHECKPOINT}...")
load_model_checkpoint(model, CHECKPOINT)
model.eval()

tokenizer_name = train_params.data.tokenizer
print(f"Loading tokenizer: {tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

inputs = tokenizer(["hi", "This is a batch"], return_tensors="pt", padding=True)
inputs = inputs.to("cuda")

print("Running inference...")
with torch.autocast(device_type="cuda", dtype=model.dtype), torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=50)

print("\n=== OUTPUT ===")
print(tokenizer.batch_decode(out, skip_special_tokens=True))
