# Visualization scripts

## Testing the dataloader:
To look at the first few samples of a dataloader, run the script below:
```
python vla_foundry/data/scripts/vis/dataloader_vis.py (--arguments-here)
```

This uses the same argument parsing `draccus.parse(config_class=TrainExperimentParams)` as `main.py`, so you can take any script in `./examples` and copy paste the arguments exactly. It will load the dataloader the exact same way that `main.py` loads it.
