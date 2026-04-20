# Examples Overview

The `examples/` directory contains copy-paste-ready bash scripts for the three
main workflows in VLA Foundry: **preprocessing**, **training**, and
**visualization**. They complement the end-to-end walkthroughs in
[Tutorials](../tutorials.md); reach for a tutorial first if you're new, then come back here
for a one-shot CLI example.

All scripts use placeholder S3 paths (`s3://your-bucket/your-path/...`) and
placeholder dataset/checkpoint names — edit them to point at your own data.

## Directory Structure

```
examples/
  training/
    llm_11m.sh                                    # 11M transformer LLM from scratch
    vlm_paligemma3b.sh                            # 3B PaliGemma-style VLM
    vlm_smolvlm_full_fromllm.sh                   # SmolVLM initialized from a pretrained LLM
    vla_diffusion_redbellpepper_paligemma2.sh     # VLA DiffusionPolicy w/ PaliGemma2 backbone
    vla_diffusion_redbellpepper_qwen_2b_thinking.sh  # VLA DiffusionPolicy w/ Qwen3-VL backbone
    diffusion_policy.sh                           # Standalone DiffusionPolicy on robotics shards
    resume.sh                                     # Resume / finetune from a checkpoint
  preprocessing/
    preprocess_robotics_data_lbm.sh               # Spartan → tar shards
    preprocess_robotics_data_lerobot.sh           # LeRobot → tar shards
  visualization/
    visualize_data.sh                             # CLI wrapper around lbm_vis.py
    visualization_params.yaml                     # Draccus config consumed by the wrapper
    README.md                                     # Usage details
  README.md                                       # (this index, but plain text)
```

Deployment scripts live separately under `vla_foundry/inference/scripts/`
(see the [deployment guide](../guides/deployment.md)).

## Training

Seven representative recipes, one per major pattern.

[See annotated training examples](training.md){ .md-button }

## Preprocessing

Two robotics preprocessing entry points; they both call
`vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py` with a
different `--type` flag.

[See annotated preprocessing examples](preprocessing.md){ .md-button }

## Visualization

`visualize_data.sh` is a thin wrapper around
`vla_foundry/data/scripts/vis/lbm_vis.py` that auto-detects S3 vs local
paths, builds the right draccus args from the dataset manifest, and forwards
them. Useful when you want to eyeball a dataset without opening a notebook.
See `examples/visualization/README.md` for flags and example invocations.
