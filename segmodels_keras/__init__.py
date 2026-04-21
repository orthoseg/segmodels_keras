import functools  # noqa: I001
import os
from typing import Any
from collections.abc import Callable

# __version__ should be defined as soon as possible
from ._version import __version__

from . import base

_KERAS_FRAMEWORK_NAME: str = "keras"
_TF_KERAS_FRAMEWORK_NAME: str = "tf.keras"

_DEFAULT_KERAS_FRAMEWORK: str = _KERAS_FRAMEWORK_NAME
_KERAS_FRAMEWORK: str | None = None
_KERAS_BACKEND: Any = None
_KERAS_LAYERS: Any = None
_KERAS_MODELS: Any = None
_KERAS_UTILS: Any = None
_KERAS_LOSSES: Any = None


def inject_global_losses(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        kwargs["losses"] = _KERAS_LOSSES
        return func(*args, **kwargs)

    return wrapper


def filter_kwargs(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        new_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["backend", "layers", "models", "utils"]
        }
        return func(*args, **new_kwargs)

    return wrapper


def framework() -> str | None:
    """Return name of Segmentation Models framework"""
    return _KERAS_FRAMEWORK


def set_framework(name: str) -> None:
    """Set framework for Segmentation Models

    Args:
        name (str): one of ``keras``, ``tf.keras``, case insensitive.

    Raises:
        ValueError: in case of incorrect framework name.
        ImportError: in case framework is not installed.

    """
    name = name.lower()

    if name == _KERAS_FRAMEWORK_NAME:
        import keras  # noqa: PLC0415
    elif name == _TF_KERAS_FRAMEWORK_NAME:
        from tensorflow import keras  # noqa: PLC0415
    else:
        raise ValueError(
            f"Not correct module name `{name}`, use `{_KERAS_FRAMEWORK_NAME}` or "
            f"`{_TF_KERAS_FRAMEWORK_NAME}`"
        )

    global _KERAS_BACKEND, _KERAS_LAYERS, _KERAS_MODELS  # noqa: PLW0603
    global _KERAS_UTILS, _KERAS_LOSSES, _KERAS_FRAMEWORK  # noqa: PLW0603

    _KERAS_FRAMEWORK = name
    _KERAS_BACKEND = keras.backend
    _KERAS_LAYERS = keras.layers
    _KERAS_MODELS = keras.models
    _KERAS_UTILS = keras.utils
    _KERAS_LOSSES = keras.losses

    # allow losses/metrics get keras submodules
    base.KerasObject.set_submodules(
        backend=keras.backend,
        layers=keras.layers,
        models=keras.models,
        utils=keras.utils,
    )


# set default framework
_framework = os.environ.get("SM_FRAMEWORK", _DEFAULT_KERAS_FRAMEWORK)
try:
    set_framework(_framework)
except ImportError:
    other = (
        _TF_KERAS_FRAMEWORK_NAME
        if _framework == _KERAS_FRAMEWORK_NAME
        else _KERAS_FRAMEWORK_NAME
    )
    set_framework(other)

print(f"Segmentation Models: using `{_KERAS_FRAMEWORK}` framework.")

# import helper modules
from . import losses, metrics, utils

# wrap segmentation models with framework modules
from .backbones.backbones_factory import Backbones
from .models.fpn import FPN
from .models.linknet import Linknet
from .models.pspnet import PSPNet
from .models.unet import Unet

get_available_backbone_names = Backbones.models_names


def get_preprocessing(name: str) -> Callable[[Any], Any]:
    return Backbones.get_preprocessing(name)


__all__ = [
    "FPN",
    "Linknet",
    "PSPNet",
    "Unet",
    "__version__",
    "framework",
    "get_available_backbone_names",
    "get_preprocessing",
    "losses",
    "metrics",
    "set_framework",
    "utils",
]
