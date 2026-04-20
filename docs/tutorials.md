# Tutorials

Interactive Jupyter notebooks that walk through end-to-end workflows in VLA Foundry. These tutorials are hands-on and beginner-friendly, designed to help you understand how to use the framework by running actual training jobs with small models and sample data.

All tutorials are located in the [`tutorials/`](https://github.com/TRI-ML/vla_foundry/tree/main/tutorials) directory of the repository.

---

## Getting Started

**Prerequisites:**
- GPU with ≥16 GB VRAM recommended
- Install the Jupyter kernel once from the repo root:
  ```bash
  bash tutorials/install_kernel.sh
  ```
- Select **Python (vla_foundry)** as your kernel when running notebooks

All notebooks are standalone and download required data automatically.

---

## Available Tutorials

### 🎯 [Training LLM, VLM, and VLA](https://github.com/TRI-ML/vla_foundry/blob/main/tutorials/training_llm_vlm_vla.ipynb)

The full three-stage training pipeline from scratch: train a 100M parameter language model on text data, add vision capabilities with image-caption training, and add action prediction with robotics data.

This is the **recommended starting point** if you're new to VLA Foundry.

---

### 🔄 [LLM & VLM Inference](https://github.com/TRI-ML/vla_foundry/blob/main/tutorials/llm_vlm_inference_tutorial.ipynb)

Load trained models and run inference: load LLM/VLM checkpoints, generate text completions and image captions, and use the processor and tokenizer APIs.

---

### 📊 [Data Visualization](https://github.com/TRI-ML/vla_foundry/blob/main/tutorials/data_visualization.ipynb)

Inspect and visualize robotics datasets: visualize camera streams, plot action trajectories, examine proprioceptive data, and debug data loading issues.

---

### 🤖 [Simulation Evaluation](https://github.com/TRI-ML/vla_foundry/blob/main/tutorials/sim_evaluation_tutorial.ipynb)

Evaluate VLA policies in simulation: set up the evaluation environment, load a trained VLA checkpoint, run rollouts in simulation, and analyze success rates and failure modes.

---

### 📦 [Adding New Datasets](https://github.com/TRI-ML/vla_foundry/blob/main/tutorials/adding_new_datasets.ipynb)

Integrate custom datasets into VLA Foundry: understand the WebDataset tar format, convert your data to VLA Foundry format, write dataset manifests, and configure dataset mixing and weighting.

---

### 🔧 [Converting Spartan Data to Tar Shards](https://github.com/TRI-ML/vla_foundry/blob/main/tutorials/convert_lbm_spartan_to_tar_shards.ipynb)

Preprocess LBM/Spartan format robotics data: convert Spartan episodes to WebDataset shards, generate dataset statistics, and create manifests for training.

---

### 🦾 [LeRobot Integration](https://github.com/TRI-ML/vla_foundry/blob/main/tutorials/lerobot.ipynb)

Work with LeRobot datasets: download datasets from the [LeRobot hub](https://huggingface.co/lerobot), convert LeRobot format to VLA Foundry format, and preprocess and train on LeRobot data.

---

## What's Next?

After completing the tutorials, check out:

- **[Examples](examples/index.md)** -- Copy-paste-ready bash scripts for production workflows
- **[Guides](guides/adding-new-models.md)** -- In-depth how-to guides for specific tasks
- **[Reference](reference/params/overview.md)** -- Detailed API documentation

---

## Troubleshooting

**Kernel not found:**
```bash
bash tutorials/install_kernel.sh
# Then restart Jupyter and select "Python (vla_foundry)"
```

**Out of memory:**
- Reduce `per_gpu_batch_size` in training commands
- Use smaller model configs (e.g., `transformer_100m.yaml` instead of larger variants)
- Close other GPU processes

**Data download fails:**
- Check your internet connection
- Some image URLs in PixelProse may be unavailable (normal -- the tutorial will retry)
- For large datasets, consider downloading outside the notebook and pointing to local paths

For more help, see the [FAQ](faq.md) or open an issue on [GitHub](https://github.com/TRI-ML/vla_foundry/issues).
