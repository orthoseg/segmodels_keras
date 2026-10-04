"""Tests on the model creation functions in segmodels_keras.models."""

import os

import numpy as np
import pytest
import six

import segmodels_keras as sm
from segmodels_keras import get_available_backbone_names, get_model
from tests.test_helper import data_dir

if sm.framework() == sm._TF_KERAS_FRAMEWORK_NAME:
    from tensorflow import keras
elif sm.framework() == sm._KERAS_FRAMEWORK_NAME:
    import keras
else:
    raise ValueError(f"Incorrect framework {sm.framework()}")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_test_backbones():
    is_full = str2bool(os.environ.get("FULL_TEST", "False"))
    if not is_full:
        return [
            "resnet34",
            "resnet50",
            "inceptionresnetv2",
            "efficientnetb0",
            "efficientnetv2s",
            "efficientnetv2s-ss",
            "efficientnetv2m-ss",
            "efficientnetv2l-ss",
        ]
    else:
        return get_available_backbone_names()


def keras_test(func):
    """Function wrapper to clean up after TensorFlow tests.
    # Arguments
        func: test function to clean up after.
    # Returns
        A function wrapping the input function.
    """

    @six.wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        keras.backend.clear_session()
        return output

    return wrapper


@pytest.mark.parametrize("backbone_name", get_test_backbones())
@pytest.mark.parametrize(
    "model_name, input_shape, encoder_weights",
    [
        ("unet", None, None),
        ("unet", None, "imagenet"),
        ("unet", (256, 256, 4), None),
        ("linknet", None, None),
        ("linknet", (256, 256, 4), None),
        ("pspnet", (384, 384, 4), None),
        ("fpn", None, None),
        ("fpn", (256, 256, 4), None),
    ],
)
@keras_test
def test_get_model(model_name, backbone_name, input_shape, encoder_weights):
    """Test all segmentation models with different backbones.

    input_shape=None means any shape (32x32 used in test), otherwise a fixed shape is
    used.
    """
    n_channels = 3 if input_shape is None else input_shape[-1]
    test_shape = (1, 32, 32, n_channels) if input_shape is None else (1, *input_shape)

    x = np.ones(test_shape)
    model = get_model(
        model_name,
        backbone_name=backbone_name,
        input_shape=input_shape or (None, None, n_channels),
        encoder_weights=encoder_weights,
    )
    y = model.predict(x)

    assert x.shape[:-1] == y.shape[:-1]


def test_get_model_invalid_name():
    """Test get_model with invalid model name."""
    with pytest.raises(ValueError, match="Unknown model name: invalid_model"):
        get_model("invalid_model")


@pytest.mark.parametrize(
    "standard_backbone, compact_backbone, expected_parameter_reduction",
    [
        ("efficientnetv2s", "efficientnetv2s-ss", 2_092_032),
        ("efficientnetv2m", "efficientnetv2m-ss", 2_331_648),
        ("efficientnetv2l", "efficientnetv2l-ss", 2_949_120),
    ],
)
@keras_test
def test_efficientnetv2_compact_skips_reduce_unet_parameters(
    standard_backbone, compact_backbone, expected_parameter_reduction
):
    input_shape = (32, 32, 3)
    standard_model = get_model(
        "unet",
        backbone_name=standard_backbone,
        input_shape=input_shape,
        encoder_weights=None,
    )
    standard_params = standard_model.count_params()
    standard_output_shape = standard_model.output_shape
    del standard_model
    keras.backend.clear_session()

    compact_model = get_model(
        "unet",
        backbone_name=compact_backbone,
        input_shape=input_shape,
        encoder_weights=None,
    )

    assert compact_model.output_shape == standard_output_shape
    assert (
        standard_params - compact_model.count_params() == expected_parameter_reduction
    )


def test_get_model_weights_notop_keras2():
    """Test loading a Keras v2 style HDF5 weight file."""
    weights_path = data_dir / "mobilenetv2+linknet_notop_k2.weights.h5"
    model = sm.Linknet("mobilenetv2", weights_notop=weights_path, encoder_weights=None)

    assert model is not None
    assert len(model.layers) > 0


def test_get_model_weights_notop_keras3():
    """Test loading a Keras v3 style HDF5 weight file."""
    weights_path = data_dir / "mobilenetv2+linknet_notop_k3.weights.h5"
    model = sm.Linknet("mobilenetv2", weights_notop=weights_path, encoder_weights=None)

    assert model is not None
    assert len(model.layers) > 0


if __name__ == "__main__":
    pytest.main([__file__])
