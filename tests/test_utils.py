import h5py
import numpy as np
import pytest

import segmodels_keras as sm
from segmodels_keras import Unet
from segmodels_keras.utils import _load_keras_v3_weights_hdf5, set_regularization

if sm.framework() == sm._TF_KERAS_FRAMEWORK_NAME:
    from tensorflow import keras
elif sm.framework() == sm._KERAS_FRAMEWORK_NAME:
    import keras
else:
    raise ValueError(f"Incorrect framework {sm.framework()}")

X1 = np.ones((1, 32, 32, 3))
Y1 = np.ones((1, 32, 32, 1))
MODEL = Unet
BACKBONE = "resnet50"
CASE = ((X1, Y1, MODEL, BACKBONE),)


def _test_regularizer(model, reg_model, x, y):

    def zero_loss(gt, pr):  # noqa: ARG001
        return pr * 0

    model.compile("Adam", loss=zero_loss, metrics=["binary_accuracy"])
    reg_model.compile("Adam", loss=zero_loss, metrics=["binary_accuracy"])

    loss_1, _ = model.test_on_batch(x, y)
    loss_2, _ = reg_model.test_on_batch(x, y)

    assert loss_1 == 0
    assert loss_2 > 0

    keras.backend.clear_session()


@pytest.mark.parametrize("case", CASE)
def test_kernel_reg(case):
    x, y, model_fn, backbone = case

    l1_reg = keras.regularizers.l1(0.1)
    model = model_fn(backbone)
    reg_model = set_regularization(model, kernel_regularizer=l1_reg)
    _test_regularizer(model, reg_model, x, y)

    l2_reg = keras.regularizers.l2(0.1)
    model = model_fn(backbone, encoder_weights=None)
    reg_model = set_regularization(model, kernel_regularizer=l2_reg)
    _test_regularizer(model, reg_model, x, y)


"""
Note:
    backbone resnet18 use BN after each conv layer --- so no bias used in these conv layers
    skip the bias regularizer test

@pytest.mark.parametrize('case', CASE)
def test_bias_reg(case):
    x, y, model_fn, backbone = case

    l1_reg = regularizers.l1(1)
    model = model_fn(backbone)
    reg_model = set_regularization(model, bias_regularizer=l1_reg)
    _test_regularizer(model, reg_model, x, y)

    l2_reg = regularizers.l2(1)
    model = model_fn(backbone)
    reg_model = set_regularization(model, bias_regularizer=l2_reg)
    _test_regularizer(model, reg_model, x, y)
"""  # noqa: E501


@pytest.mark.parametrize("case", CASE)
def test_bn_reg(case):
    x, y, model_fn, backbone = case

    l1_reg = keras.regularizers.l1(1)
    model = model_fn(backbone)
    reg_model = set_regularization(model, gamma_regularizer=l1_reg)
    _test_regularizer(model, reg_model, x, y)

    model = model_fn(backbone)
    reg_model = set_regularization(model, beta_regularizer=l1_reg)
    _test_regularizer(model, reg_model, x, y)

    l2_reg = keras.regularizers.l2(1)
    model = model_fn(backbone)
    reg_model = set_regularization(model, gamma_regularizer=l2_reg)
    _test_regularizer(model, reg_model, x, y)

    model = model_fn(backbone)
    reg_model = set_regularization(model, beta_regularizer=l2_reg)
    _test_regularizer(model, reg_model, x, y)


@pytest.mark.parametrize("case", CASE)
@pytest.mark.xfail(reason="test_activity_reg seems to fail, not sure why")
def test_activity_reg(case):
    x, y, model_fn, backbone = case

    l2_reg = keras.regularizers.l2(1)
    model = model_fn(backbone)
    reg_model = set_regularization(model, activity_regularizer=l2_reg)
    _test_regularizer(model, reg_model, x, y)


@pytest.mark.parametrize("decoder", ["fpn", "linknet", "unet", "pspnet"])
def test_save_model_weights_notop(tmp_path, decoder):
    backbone_name = "mobilenetv2"
    if decoder == "pspnet":
        model = sm.PSPNet(backbone_name, encoder_weights=None)
    elif decoder == "fpn":
        model = sm.FPN(backbone_name, encoder_weights=None)
    elif decoder == "linknet":
        model = sm.Linknet(backbone_name, encoder_weights=None)
    elif decoder == "unet":
        model = sm.Unet(backbone_name, encoder_weights=None)
    else:
        raise ValueError(f"Incorrect decoder {decoder}")

    # Test saving the model weights without the top layers
    path = tmp_path / f"{decoder}.weights.h5"
    sm.utils.save_model_weights_notop(model, decoder=decoder, path=path)

    # Test loading the model weights without the top layers again
    assert path.exists()
    if decoder == "pspnet":
        model = sm.PSPNet(backbone_name, weights_notop=path, freeze_notop=True)
    elif decoder == "fpn":
        model = sm.FPN(backbone_name, weights_notop=path, freeze_notop=True)
    elif decoder == "linknet":
        model = sm.Linknet(backbone_name, weights_notop=path, freeze_notop=True)
    elif decoder == "unet":
        model = sm.Unet(backbone_name, weights_notop=path, freeze_notop=True)
    else:
        raise ValueError(f"Incorrect decoder {decoder}")


def test_save_model_weights_notop_invalid_decoder():
    model = keras.models.Model()
    with pytest.raises(ValueError, match="Decoder should be one of"):
        sm.utils.save_model_weights_notop(
            model, decoder="invalid_decoder", path="path/to/weights.h5"
        )


def test_load_keras_v3_style_weights_hdf5(tmp_path):
    """Normal case: distinct weight-shape signatures per layer."""
    inputs = keras.layers.Input(shape=(4,), name="input")
    x = keras.layers.Dense(3, use_bias=True, name="dense")(inputs)
    x = keras.layers.BatchNormalization(scale=False, name="batch_norm")(x)
    outputs = keras.layers.Dense(2, use_bias=False, name="head")(x)
    model = keras.models.Model(inputs, outputs, name="functional")

    expected_weights = [weight.copy() for weight in model.get_weights()]
    path = tmp_path / "keras_v3_style.weights.h5"

    with h5py.File(path, "w") as handle:
        layers_group = handle.create_group("layers")
        handle.create_group("vars").attrs["name"] = model.name

        for layer in model.layers:
            layer_group = layers_group.create_group(layer.name)
            vars_group = layer_group.create_group("vars")
            for index, value in enumerate(layer.get_weights()):
                vars_group.create_dataset(str(index), data=value)

    zero_weights = [np.zeros_like(weight) for weight in expected_weights]
    model.set_weights(zero_weights)

    _load_keras_v3_weights_hdf5(model, path)

    for actual, expected in zip(model.get_weights(), expected_weights, strict=True):
        np.testing.assert_allclose(actual, expected)


def test_load_keras_v3_style_weights_hdf5_duplicate_signatures(tmp_path):
    """Layers with identical weight-shape signatures are matched in save order.

    This gave an error for Keras v3 HDF5 weight files of an inceptionresnetv2 backbone.
    """
    # All three Dense layers share the same shape signature: [(4, 3)].
    # The fallback must pop each candidate in file-write order so that
    # branch_a, branch_b, and branch_c each receive their own distinct values.
    inputs = keras.layers.Input(shape=(4,), name="input")
    branch_a = keras.layers.Dense(3, use_bias=False, name="branch_a")(inputs)
    branch_b = keras.layers.Dense(3, use_bias=False, name="branch_b")(inputs)
    branch_c = keras.layers.Dense(3, use_bias=False, name="branch_c")(inputs)
    concat_axis = 1
    outputs = keras.layers.Concatenate(axis=concat_axis, name="concat")(
        [branch_a, branch_b, branch_c]
    )
    model = keras.models.Model(inputs, outputs, name="functional")

    # Assign distinct sentinel values to each dense layer so we can verify
    # that the right values end up in the right layer after loading.
    dense_layers = [model.get_layer(n) for n in ("branch_a", "branch_b", "branch_c")]
    original_values = []
    for i, layer in enumerate(dense_layers):
        sentinel = np.full((4, 3), fill_value=float(i + 1), dtype=np.float32)
        layer.set_weights([sentinel])
        original_values.append(sentinel.copy())

    path = tmp_path / "dup_signatures.weights.h5"
    with h5py.File(path, "w") as handle:
        layers_group = handle.create_group("layers")
        handle.create_group("vars").attrs["name"] = model.name
        for layer in model.layers:
            layer_group = layers_group.create_group(layer.name)
            vars_group = layer_group.create_group("vars")
            for index, value in enumerate(layer.get_weights()):
                vars_group.create_dataset(str(index), data=value)

    # Zero out all weights before loading.
    for layer in dense_layers:
        layer.set_weights([np.zeros((4, 3), dtype=np.float32)])

    _load_keras_v3_weights_hdf5(model, path)

    for layer, expected in zip(dense_layers, original_values, strict=True):
        actual = layer.get_weights()[0]
        np.testing.assert_allclose(
            actual,
            expected,
            err_msg=f"Layer '{layer.name}' got wrong weights after loading.",
        )


if __name__ == "__main__":
    pytest.main([__file__])
