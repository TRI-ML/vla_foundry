import logging
import typing
from collections.abc import Sequence as SequenceType
from dataclasses import fields, is_dataclass
from typing import Any, get_args, get_origin

from draccus.choice_types import CHOICE_TYPE_KEY
from draccus.parsers.decoding import decode_choice_class
from draccus.utils import is_choice_type

# Field migrations: mapping of (path_prefix, old_field_name) -> new_field_name
# path_prefix is a tuple of field names leading to the field, e.g., ("image",) for augmentation.image
FIELD_MIGRATIONS = {
    (("image",), "random_crop"): "crop",  # When loaded as nested field in DataAugmentationParams
    ((), "random_crop"): "crop",  # When ImageAugmentationParams is loaded directly
}


def _apply_field_migrations(raw_value: Any, path: SequenceType[str]) -> Any:
    """Apply field migrations to rename deprecated fields."""
    if not isinstance(raw_value, dict):
        return raw_value

    result = raw_value.copy()
    for (path_prefix, old_name), new_name in FIELD_MIGRATIONS.items():
        if path == path_prefix and old_name in result and new_name not in result:
            logging.warning(f"Migrating deprecated field '{old_name}' to '{new_name}' at path {'.'.join(path)}")
            result[new_name] = result.pop(old_name)

    return result


def _strip_unknown_keys(raw_value: Any, cls: type[Any], path: SequenceType[str]) -> Any:
    """Ignore keys that are not in the dataclass definition while decoding."""
    target_cls = _resolve_dataclass(cls)
    if target_cls is None or not isinstance(raw_value, dict):
        return raw_value

    # Apply field migrations before stripping unknown keys
    raw_value = _apply_field_migrations(raw_value, path)

    # For Choice types, determine the actual subclass from the 'type' field
    if is_choice_type(target_cls) and CHOICE_TYPE_KEY in raw_value:
        choice_type = raw_value.get(CHOICE_TYPE_KEY)
        try:
            actual_cls = target_cls.get_choice_class(choice_type)
            # Use the actual subclass for field validation
            target_cls = actual_cls
        except (KeyError, AttributeError):
            # If we can't resolve, use the base class
            pass

    allowed_fields = {field.name for field in fields(target_cls)}
    # Always allow the CHOICE_TYPE_KEY through
    allowed_fields.add(CHOICE_TYPE_KEY)

    cleaned = {key: value for key, value in raw_value.items() if key in allowed_fields}
    removed_fields = [key for key in raw_value if key not in allowed_fields]
    if removed_fields:
        readable_path = ".".join(path) if path else target_cls.__name__
        logging.warning(
            f"Ignoring unknown config fields {', '.join(sorted(removed_fields))} while decoding {readable_path}."
        )

    # Recursively clean nested dataclass fields
    for field_info in fields(target_cls):
        field_name = field_info.name
        if field_name in cleaned and isinstance(cleaned[field_name], dict):
            field_type = field_info.type
            # Resolve the actual type
            field_cls = _resolve_dataclass(field_type)
            if field_cls is not None:
                # Recursively strip unknown keys from nested dataclass
                cleaned[field_name] = _strip_unknown_keys(cleaned[field_name], field_type, (*path, field_name))
            elif get_origin(field_type) is typing.Union:
                # For Union types, strip keys not valid for any member of the union
                nested_val = cleaned[field_name]
                union_allowed = {CHOICE_TYPE_KEY}
                for union_arg in get_args(field_type):
                    arg_cls = _resolve_dataclass(union_arg)
                    if arg_cls is not None:
                        union_allowed.update(f.name for f in fields(arg_cls))
                removed = [k for k in nested_val if k not in union_allowed]
                if removed:
                    readable_path = ".".join((*path, field_name))
                    logging.warning(
                        f"Ignoring unknown config fields {', '.join(sorted(removed))} while decoding {readable_path}."
                    )
                cleaned[field_name] = {k: v for k, v in nested_val.items() if k in union_allowed}

    return cleaned


def _resolve_dataclass(cls: type[Any]) -> type[Any] | None:
    origin = get_origin(cls)
    target_cls = cls if origin is None else origin

    if isinstance(target_cls, type) and is_dataclass(target_cls):
        return target_cls
    return None


def _decode_choice_base_params(cls: type["BaseParams"], raw_value: Any, path: SequenceType[str]):  # noqa: F821 BaseParams would be circular if imported
    """Handle BaseParams that also behave as Choice types (e.g., DataParams)."""
    if not isinstance(raw_value, dict):
        return decode_choice_class(cls, raw_value, path)

    choice_type = raw_value.get(CHOICE_TYPE_KEY, cls.default_choice_name())
    if choice_type is None:
        return decode_choice_class(cls, raw_value, path)

    try:
        subcls = cls.get_choice_class(choice_type)
    except KeyError:
        return decode_choice_class(cls, raw_value, path)

    payload = {k: v for k, v in raw_value.items() if k != CHOICE_TYPE_KEY}
    cleaned_payload = _strip_unknown_keys(payload, subcls, (*path, choice_type))
    cleaned_payload = cleaned_payload.copy()
    cleaned_payload[CHOICE_TYPE_KEY] = choice_type
    return decode_choice_class(cls, cleaned_payload, path)
