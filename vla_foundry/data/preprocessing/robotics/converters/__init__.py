from vla_foundry.data.preprocessing.robotics.converters.base import BaseRoboticsConverter
from vla_foundry.data.preprocessing.robotics.preprocess_params import PreprocessParams


def get_converter(cfg: PreprocessParams) -> BaseRoboticsConverter:
    if cfg.type == "spartan":
        from vla_foundry.data.preprocessing.robotics.converters.spartan import SpartanConverter

        return SpartanConverter(cfg)
    elif cfg.type == "lerobot":
        from vla_foundry.data.preprocessing.robotics.converters.lerobot import LeRobotConverter

        return LeRobotConverter(cfg)
    else:
        raise ValueError(f"Unsupported source type: {cfg.type}")
