import json
from collections.abc import Mapping, Sequence
from collections.abc import Sequence as SequenceType
from dataclasses import dataclass, fields
from typing import Any

import draccus
from draccus.cfgparsing import load_config
from draccus.parsers.decoding import (
    decode as draccus_decode,
)
from draccus.parsers.decoding import (
    decode_dataclass,
)

from vla_foundry.file_utils import copy_to_temp_file
from vla_foundry.params.params_utils import (
    _decode_choice_base_params,
    _resolve_dataclass,
    _strip_unknown_keys,
    is_choice_type,
)


@dataclass(frozen=True)
class BaseParams:
    """
    BaseParams is the base class for all parameters. Other params classes inherit from it.
    """

    def __post_init__(self):
        pass

    def __iter__(self):
        """Make the class iterable, yielding (field_name, value) pairs."""
        for field_info in fields(self):
            field_name = field_info.name
            field_value = getattr(self, field_name)

            # Support nested BaseParams objects
            if isinstance(field_value, BaseParams):
                for nested_name, nested_value in field_value:
                    yield f"{field_name}.{nested_name}", nested_value
            else:
                yield field_name, field_value

    def get(self, key, default=None):
        return getattr(self, key, default)

    def init_shared_attributes(self, cfg):
        for field_info in fields(self):
            field_value = getattr(self, field_info.name)
            self._init_child_shared_attributes(field_value, cfg)

    def _init_child_shared_attributes(self, value, cfg):
        if isinstance(value, BaseParams):
            value.init_shared_attributes(cfg)
        elif isinstance(value, Mapping):
            for item in value.values():
                self._init_child_shared_attributes(item, cfg)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                self._init_child_shared_attributes(item, cfg)

    @classmethod
    def from_file(cls, file_path):
        # Load the YAML using draccus's load_config (supports !include)
        if file_path.startswith("s3"):
            with copy_to_temp_file(file_path) as temp_path, open(temp_path) as f:
                data_dict = load_config(f, file=temp_path)
        else:
            with open(file_path) as f:
                data_dict = load_config(f, file=file_path)

        # Use from_dict which handles unknown key stripping
        return cls.from_dict(data_dict)

    @classmethod
    def from_dict(cls, dict_data):
        # Strip unknown keys before decoding
        cleaned_dict = _strip_unknown_keys(dict_data, cls, ())
        cfg_new = draccus.decode(cls, cleaned_dict)
        return cfg_new


# Register a special decoder for BaseParams that handles unknown keys without failing
@draccus_decode.register(BaseParams, include_subclasses=False)
def _decode_base_params(cls: type[BaseParams], raw_value: Any, path: SequenceType[str]):
    target_cls = _resolve_dataclass(cls)
    if target_cls is not None and is_choice_type(target_cls):
        return _decode_choice_base_params(cls, raw_value, path)

    if isinstance(raw_value, dict):
        raw_value = _strip_unknown_keys(raw_value, cls, path)
    return decode_dataclass(cls, raw_value, path)


# Make classes that define to_dict method JSON-serializable
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


json._default_encoder = CustomJSONEncoder()
