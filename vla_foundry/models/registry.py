"""
Central registry for deep learning models, batch handlers, and FSDP blocks.

This module provides a decorator-based registration system that allows models to be
self-contained and easily enabled/disabled without modifying central configuration files.

Example usage:
    # In models/my_model/__init__.py
    from vla_foundry.models.registry import register_model, register_batch_handler

    @register_model("my_model")
    def create_my_model(model_params, load_pretrained=True):
        return MyModel(model_params)

    @register_batch_handler("my_model")
    class MyModelBatchHandler(BatchHandler):
        ...
"""

from collections.abc import Callable

import torch.nn as nn

from vla_foundry.params.model_params import ModelParams

# Global registries
_MODEL_REGISTRY: dict[str, Callable] = {}
_BATCH_HANDLER_REGISTRY: dict[str, type] = {}


def register_model(model_type: str):
    """
    Decorator to register a model creation function.

    Args:
        model_type: Unique identifier for the model type (e.g., "diffusion_policy")

    Example:
        @register_model("my_model")
        def create_my_model(model_params: ModelParams, load_pretrained: bool = True):
            return MyModel(model_params)
    """

    def decorator(create_fn: Callable):
        if model_type in _MODEL_REGISTRY:
            raise ValueError(f"Model type '{model_type}' is already registered")
        _MODEL_REGISTRY[model_type] = create_fn
        return create_fn

    return decorator


def register_batch_handler(model_type: str):
    """
    Decorator to register a batch handler class.

    Args:
        model_type: Model type identifier matching the registered model

    Example:
        @register_batch_handler("my_model")
        class MyModelBatchHandler(BatchHandler):
            ...
    """

    def decorator(handler_cls: type):
        if model_type in _BATCH_HANDLER_REGISTRY:
            raise ValueError(f"Batch handler for '{model_type}' is already registered")
        _BATCH_HANDLER_REGISTRY[model_type] = handler_cls
        return handler_cls

    return decorator


def create_model(model_params: ModelParams, load_pretrained: bool = True) -> nn.Module:
    """
    Create a model from registered model types.

    Args:
        model_params: Model configuration parameters
        load_pretrained: If True, download pretrained weights

    Returns:
        Instantiated model

    Raises:
        ValueError: If model type is not registered
    """
    model_type = model_params.type
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model type '{model_type}' is not registered. Available models: {list(_MODEL_REGISTRY.keys())}"
        )

    create_fn = _MODEL_REGISTRY[model_type]
    return create_fn(model_params, load_pretrained)


def create_batch_handler(model_type: str):
    """
    Create a batch handler for the specified model type.

    Args:
        model_type: The type of model

    Returns:
        BatchHandler instance

    Raises:
        ValueError: If batch handler is not registered for this model type
    """
    if model_type not in _BATCH_HANDLER_REGISTRY:
        raise ValueError(
            f"Batch handler for model type '{model_type}' is not registered. "
            f"Available handlers: {list(_BATCH_HANDLER_REGISTRY.keys())}"
        )

    handler_cls = _BATCH_HANDLER_REGISTRY[model_type]
    return handler_cls()


def list_registered_models() -> list[str]:
    """Return list of all registered model types."""
    return list(_MODEL_REGISTRY.keys())


def list_registered_batch_handlers() -> list[str]:
    """Return list of model types with registered batch handlers."""
    return list(_BATCH_HANDLER_REGISTRY.keys())


def is_model_registered(model_type: str) -> bool:
    """Check if a model type is registered."""
    return model_type in _MODEL_REGISTRY
