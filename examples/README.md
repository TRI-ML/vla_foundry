# Examples

Copy-paste-ready shell scripts that exercise the main VLA Foundry workflows.
These are complements to the more detailed walkthroughs in `tutorials/`;
reach for a tutorial first if you're new to a workflow, and come back here
for a one-shot CLI example.

All scripts use placeholder S3 paths (`s3://your-bucket/your-path/...`) and
placeholder dataset/checkpoint names — edit them to point at your own data.

## Training (`examples/training/`)

One representative script per major recipe:

| Script | Recipe |
|---|---|
| `llm_11m.sh` | Train a 11M parameter transformer LLM from scratch (tokenized text, 3 GPUs, FSDP). |
| `vlm_paligemma3b.sh` | Train a 3B PaliGemma-style VLM on image-caption data. |
| `vlm_smolvlm_full_fromllm.sh` | Train a SmolVLM-style VLM initialized from a pretrained LLM checkpoint (shows the `--model.transformer.resume_from_checkpoint` + `--model.transformer.resume_weights_only` pattern). |
| `vla_diffusion_redbellpepper_paligemma2.sh` | Train a VLA DiffusionPolicy with a PaliGemma2 backbone on the `BimanualPutRedBellPepperInBin` task. |
| `vla_diffusion_redbellpepper_qwen_2b_thinking.sh` | Same VLA recipe, but with a Qwen3-VL-2B-Thinking backbone. |
| `diffusion_policy.sh` | Train a standalone DiffusionPolicy (no VLM) on preprocessed robotics shards. |
| `resume.sh` | Resume training from a saved checkpoint (`--model.resume_from_checkpoint` + `--model.resume_weights_only`). |

## Preprocessing (`examples/preprocessing/`)

| Script | Recipe |
|---|---|
| `preprocess_robotics_data_lbm.sh` | Convert raw Spartan-format LBM episodes into webdataset tar shards for training. |
| `preprocess_robotics_data_lerobot.sh` | Convert a LeRobot-format HuggingFace dataset into webdataset tar shards. |

See also `vla_foundry/data/preprocessing/README.md` for Ray-cluster setup
and the full preprocessing pipeline.

## Visualization (`examples/visualization/`)

| File | Purpose |
|---|---|
| `visualize_data.sh` | Wrapper around `vla_foundry/data/scripts/vis/lbm_vis.py` that auto-detects S3 vs local dataset paths and forwards the right draccus args. |
| `visualization_params.yaml` | Draccus config preset consumed by `visualize_data.sh`. |
| `README.md` | Usage details and flags. |

## See Also

- `tutorials/training_llm_vlm_vla.ipynb` — end-to-end LLM → VLM → VLA walkthrough
- `tutorials/lerobot.ipynb` — LeRobot preprocessing + robotics training
- `tutorials/data_visualization.ipynb` — inline Rerun visualization walkthrough
- `tutorials/sim_evaluation_tutorial.ipynb` — simulation evaluation with `lbm_eval`
- `vla_foundry/inference/scripts/README.md` — gRPC policy server scripts
- `sagemaker/README.md` — launching training on AWS SageMaker
