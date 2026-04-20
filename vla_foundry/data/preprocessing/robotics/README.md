# Robotics Data Preprocessing

This README specifically discusses the robotics data preprocessing file [preprocess_robotics_to_tar.py](../preprocess_robotics_to_tar.py). For a more general README, please see the [README in the scripts folder](./../README.md).

Because we may want robotics data from different sources, we create a unified structure and a unified script to handle the preprocessing. This lets us avoid having to reimplement certain functionalities like ray parallelism. Instead, these functionalities are shared, and the dataset-specific processing logic is moved to the individual converters inside [converters/](converters/).

## 1. Converters
To select which dataset format to use, you can use the `--type` argument. This will route your script to the correct converter class for your dataset. You can then supply class-specific parameters directly.

All converter classes inherit from the base class `BaseRoboticsConverter`. The [preprocess_robotics_to_tar.py](../preprocess_robotics_to_tar.py) interfaces with converter objects by calling methods such as `discover_episodes` and `process_episode`. 

### 1.1 Adding a New Dataset Source
To support a new dataset source, create a new class inside `converters`, register your class in `converters/__init__.py`, add a custom `PreprocessParams` (see [1.2 Preprocessing Parameters](#12-preprocessing-parameters)), then define the following methods (`converters/base.py` also provides some docstring guides for how to populate these methods).
- `discover_episodes`: Given a list of paths, return a list of full paths for each episode.
- `process_episode`: This is pre-filled in `base.py` with the logic to extract the necessary fields, as well as the parallelism for uploading. The functioning of this method depends on the other methods listed below. For most cases, you probably will not need to touch this specific function, and it should work properly once all the other methods are defined. 
- `load_episode_data`: Takes in `episode_path` and reads it, then extracts and returns `episode_data`. This `episode_data` is what is passed to the other functions to extract from, so this `episode_data` return format can be any format you wish.
    - `get_episode_length`: Takes in `episode_data`. Returns length of episode.
    - `extract_camera_data`: Takes in `episode_data` then extracts out the image data and returns a dict. The result is passed as an argument to `extract_sample_data` later on.
    - `extract_lowdim_data`: Same as above but for lowdim.
    - `extract_intrinsics_extrinsics_data`: Same as above but for intrinsics and extrinsics. Can return `None` if these are not present.
    - `extract_metadata_data`: Same as above but for metadata. 
- `extract_sample_data`: The previously defined `extract_*_data` functions contain data for all the timesteps. This function takes in `anchor_timesteps` and uses it to extract the data relevant to the current frame. It returns `sample_images`, `sample_lowdim`, `sample_metadata`, and `language_instructions`, which correspond to the fields in our output tar shards.

For examples on how to fill out these methods above, see [converters/spartan.py](converters/spartan.py) and [converters/lerobot.py](converters/lerobot.py)


### 1.2 Preprocessing Parameters
The `preprocess_robotics_to_tar.py` file reads parameters from the `PreprocessParams` class from [preprocess_params.py](preprocess_params.py). This inherits from the `BaseParams` class, and parsing is done with the `draccus` parser. These `PreprocessParams` is passed to the `BaseRoboticsConverter` initializer, and can be accessed by calling `self.cfg`. 

The `PreprocessParams` has multiple subclasses such as `SpartanPreprocessParams` or `LeRobotPreprocessParams`, each with their own class-specific attributes. Users can specify the appropriate class to use the `--type` flag.


## 2. Parallelism
We use Ray for parallelism. For a guide on how to use Ray, see the [README in the scripts folder](./../README.md). Luckily, these are already defined in `preprocess_robotics_to_tar.py` and in the `BaseRoboticsConverter` class, so for adding new data sources, you will not need to deal with Ray and can instead focus on defining the preprocessing logic in the converters classes. 

Within each Ray node, we also use ThreadPoolExecutor to speed up the processing and uploading. Once again, this has already been handled in the `BaseRoboticsConverter` class.

The preprocessing happens through two Ray phases. In the first phase, we read the path to a raw unprocessed episode, then create one tar shard per timestep per episode. This phase is where most of the preprocessing logic is done. In the second phase, we gather all the timestep tars from the previous phase, and we shard them into grouped shards according to the parameter `samples_per_shard`. Both phases are done with Ray remote functions.


## 3. Statistics Computation
Statistics computation is done in [preprocess_statistics.py](preprocess_statistics.py). The `preprocess_robotics_to_tar.py` creates a Ray actor object, which is passed as an argument when the converter classes run the preprocessing. This class contains running tallies of various fields, together with running percentages. At the end of everything, we call `statistics_ray_actor.get_statistics.remote()`, which computes and returns the final statistics. These are uploaded as `stats.json` in the `output_dir/shards` folder.


## 4. Input and Output Structure
The output format is as follows:
```
output_directory/
├── episode_{X}_frame_{Y}.tar     # Intermediate tar shards
├── [...]
└── shards
    ├── shard_000000.tar          # Training data shards
    ├── shard_000001.tar
    ├── manifest.jsonl            # Tallies for shard indices and counts
    └── stats.json                # Dataset statistics
  ```

**Inside each shard (e.g., shard_00000.tar):**
```
├── (unique_id_1).lowdim.npz
├── (unique_id_1).camera_name_{1,2,...n}.jpg
├── (unique_id_1).language_instructions.json
├── (unique_id_1).metadata.json
└── ...
```
- `lowdim.npz` is a dict.
    - It MUST contain the following keys
        - `past_mask`
        - `future_mask`
    - It will also contain the keys that will be used to construct the actions, proprioceptions, intrinsics, and extrinsics.
- `language_instructions.json` should be a dict with keys in set ["original", "randomized", "verbose", "alternative"]
- The key names in the `lowdim.npz` dict should also exist as keys in `stats.json`


We don't have any restrictions on the input format. As long as the converter classes can handle the preprocessing for its corresponding data sources, then any input format is fine.

