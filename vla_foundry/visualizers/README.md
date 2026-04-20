# VLA Foundry Visualizers

This directory contains the visualization tools for the VLA Foundry project. These tools allow you to log and visualize data such as images, 3D points, trajectories, and scalar values during training or debugging. The visualizers are designed to be modular and support multiple backends.

## Example Usage

To see an example of how to use the interface, you can run

```
VISUALIZER=rerun uv run example_usage.py
```
or
```
VISUALIZER=wandb WANDB_PROJECT=my_project uv run example_usage.py
```

If you run simply
```
uv run example_usage.py
```
visualization will default to `disabled` if no backend is selected.

---

## Notes
- The `visualizer.py` facade automatically disables visualization if no backend is selected.
- Use the `VISUALIZER` environment variable to control the backend selection.
- For more advanced use cases, refer to the backend-specific files (`rerun_backend.py`, `wandb_backend.py`, etc.).
- By default, the rerun backend is viewable view the browser at a url such as
  ```http://localhost:9090/?url=rerun%2Bhttp://127.0.0.1:9876/proxy```. Note that
  you must forward ports 9090 and 9876.

## Available Visualizers

### 1. **Visualizer Facade**
   - **File**: `visualizer.py`
   - **Purpose**: Provides a unified interface for all visualizers.
   - **Use Cases**:
     - Abstracting backend-specific details.
     - Automatically selecting the appropriate backend based on the `VISUALIZER` environment variable.
   - **Key Features**:
     - Supports multiple backends (`rerun` or `disabled`).
     - Automatically disables visualization if no backend is selected. Defaults to `disabled`.


### 2. **Rerun Visualizer**
   - **File**: `rerun_backend.py`
   - **Purpose**: Logs data to the [Rerun.io](https://rerun.io/) visualization platform.
   - **Use Cases**:
     - Visualizing 3D points, trajectories, and rigid transforms.
     - Logging images and scalar values.
     - Debugging robot arm poses and action predictions.
     - Useful for local visualization of realtime inference, etc
   - **Key Features**:
     - Supports hierarchical logging paths.
     - Provides decorators for logging images and robot arm poses.

### 3. **WandB Visualizer**
   - **File**: `wandb_backend.py`
   - **Purpose**: Logs data to the [Weights & Biases (wandb)](https://wandb.ai/) experiment tracking platform.
   - **Use Cases**:
     - Logging scalar metrics, images, and 3D data for experiment tracking.
     - Visualizing training progress and results in the WandB dashboard.
     - Sharing experiment results and visualizations with collaborators.
   - **Key Features**:
     - Supports logging images, scalars, 3D points, line strips, poses, and text.
     - Integrates with the unified visualizer facade and can be selected via `VISUALIZER=wandb`.
     - Automatically creates a new run in the `vla_foundry` project on WandB.

---

## How to Use the Visualizers

### 1. **Setup**
Install the visualization dependency group to get both backends:

```bash
uv sync --group visualization
```

This installs `rerun-sdk`, `plotly`, and other visualization dependencies. The `rerun-sdk` is also available in the `tutorials` group (`uv sync --group tutorials`).

### 2. **Initialization**
Use the `visualizer.py` facade to initialize the visualizer. The backend is automatically selected based on the `VISUALIZER` environment variable.

```python
from vla_foundry.visualizers import visualizer as vz

# Initialize the visualizer
vz.init(run_name="example_run", add_rank_to_run=True)
```

- **Environment Variable**: Set `VISUALIZER` to one of the following:
  - `rerun`: Use the Rerun.io backend.
  - `wandb`: Use the Weights & Biases backend (requires `WANDB_PROJECT` to be set).
  - `disabled`: Disable visualization (default if unset).

### 3. **Logging Data**
The `visualizer.py` facade provides functions for logging various types of data:

#### Log an Image
```python
import numpy as np

image = np.random.rand(100, 100, 3)  # Example image
vz.log_images("example/image1", image)
```

#### Log Multiple Images
```python
images = {
    "example/image2": np.random.rand(100, 100, 3),
    "example/image3": np.random.rand(100, 100, 3),
}
vz.log_images("example/images", images)
```

#### Log Scalars
```python
vz.log_scalar("example/scalar1", 42.0)
```

#### Log 3D Points
```python
points = np.random.rand(10, 3)  # Example 3D points
vz.log_points3d("example/points", points)
```

#### Log Trajectories
```python
trajectory = np.random.rand(20, 3)  # Example trajectory
vz.log_trajectory("example/trajectory", trajectory)
```

#### Log Rigid Transforms
```python
from pydrake.math import RigidTransform

transform = RigidTransform()  # Example rigid transform
vz.log_rigid_transform("example/transform", transform)
```

### Sparse logging and runtime enable/disable

The Visualizer facade supports reducing log volume during eval without removing log calls.


#### Enable / disable logging at runtime, example:

```python
from vla_foundry.visualizers import visualizer as vz

vz.disable()  # subsequent log_* calls become no-ops
vz.enable()   # resume logging
```

#### Log every N calls per name
All log_* functions accept an optional n (alias: every_n) argument. Counters are tracked per (log type, path).

```
vz.log_images("observation_images", images, n=100)  # logs on call 1, 100, 200, ...
vz.log_scalar("metrics/loss", loss, n=50)           # logs on call 1, 50, 100, ...
```

#### Tests:

W&B:

```
VISUALIZER=wandb WANDB_PROJECT=my_project uv run python - <<'PY'
from vla_foundry.visualizers import visualizer as vz
import numpy as np
vz.init(run_name="sparse_logging_smoke")

vz.enable()
for i in range(1, 11):
    vz.log_scalar("every_n/n3", i, n=3)
    vz.log_scalar("every_n/n4", i, n=4)
    vz.log_images("every_n/img_n5", np.random.rand(64,64,3), n=5)

for i in range(1, 11):
    vz.enable() if (i % 2 == 0) else vz.disable()
    vz.log_scalar("toggle/only_even_iterations", i)

vz.shutdown()
PY
```

Rerun:

```
VISUALIZER=rerun uv run python - <<'PY'
from vla_foundry.visualizers import visualizer as vz
import numpy as np
vz.init(run_name="sparse_rerun_smoke")
vz.enable()
for i in range(1, 11):
    vz.log_scalar("every_n/n3", i, n=3)
    vz.log_images("every_n/img_n5", np.random.rand(64,64,3), n=5)
vz.shutdown()
PY
```


### 4. **Shutdown**
The visualizer will automatically shut down at the end of the program. However, you can explicitly call `shutdown` if needed:
```python
vz.shutdown()
```

---

## Choosing a Backend

### Rerun Backend
- **Best For**: Advanced 3D visualization, hierarchical logging, and debugging robot arm poses.
- **Setup**: `uv sync --group visualization` (or `--group tutorials`)
- **Environment Variable**: `VISUALIZER=rerun`

### WandB Backend
- **Best For**: Experiment tracking, logging scalar metrics, and visualizing 3D data.
- **Setup**: `uv sync --group visualization` — wandb is a base dependency, but plotly (used for 3D plots) is in the visualization group.
- **Environment Variables**:
  - `VISUALIZER=wandb` — selects the wandb backend.
  - `WANDB_PROJECT` — **required**. The wandb project to log to (e.g. `export WANDB_PROJECT=my_project`). The backend will raise an error if this is not set.

### Disabled
- **Best For**: Running without visualization (e.g., in production or testing environments).
- **Setup**: No additional setup required.
- **Environment Variable**: `VISUALIZER=disabled`

## Planned Backends:
- **gradio**: Interactive web-based visualizations.

---

## Rank Variable Usage in Visualization

TODO: This library has not been tested in a multi-GPU setting yet.

### Purpose of the Rank Variable
The `rank` variable is used to namespace logs in multi-node or multi-process environments. This ensures that logs from different processes or nodes do not overwrite each other and can be easily distinguished during debugging or visualization.

### How It Works
The rank is detected automatically from common environment variables such as:
- `RANK` (used in PyTorch distributed training)
- `SLURM_PROCID` (used in SLURM job scheduling)
- `LOCAL_RANK` (used in local multi-GPU setups)
- `OMPI_COMM_WORLD_RANK` (used in MPI environments)

If no rank is detected, the visualizer assumes a single-process setup and does not add a rank prefix.

### Examples

#### Single-Process Logging
If no rank is detected, logs are stored without a rank prefix:
```python
vz.init(run_name="example_run")
vz.log_scalar("metrics/loss", 0.123)
# Logs to: "metrics/loss"
```

#### Multi-Process Logging with `torchrun`
In a multi-process setup using `torchrun`, the `RANK` environment variable is automatically set:
```bash
torchrun --nproc_per_node=4 example_script.py
```

In your script:
```python
import os
import numpy as np
import vla_foundry.visualizers as vz

vz.init(run_name="example_run", add_rank_to_run=True)
vz.log_scalar("metrics/loss", 0.123)
# Logs to: "r{rank}/metrics/loss", where {rank} is the process rank
```

This ensures that logs from each process are stored separately.

#### Multi-GPU Training Loop Example with Optional Rank 0 Logging
When using PyTorch for distributed training across multiple GPUs, you can integrate the visualizer as follows. This example includes an option to log only from rank 0:

```python
import torch
import numpy as np
import vla_foundry.visualizers as vz

# Detect rank from environment variables
rank = int(os.environ.get("RANK", 0))

# Initialize the visualizer with rank-aware logging
vz.init(run_name="multi_gpu_training", add_rank_to_run=True)

# Set this flag to True to log only from rank 0
log_only_rank_0 = True

# Example training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Perform training step
        loss = compute_loss(batch)

        # Log only from rank 0 if the flag is set
        if not log_only_rank_0 or rank == 0:
            vz.log_scalar("metrics/loss", loss.item())

# Shutdown the visualizer
vz.shutdown()
```

### Notes
- The `log_only_rank_0` flag allows you to control whether all ranks log or only rank 0 logs. This can help reduce redundant logs in multi-GPU setups.
- The rank prefix is optional and can be disabled by setting `add_rank_to_run=False` during initialization.
- The rank detection logic is implemented in the `_detect_rank_prefix` function in `visualizer.py`.

---

