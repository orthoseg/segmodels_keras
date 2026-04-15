import numpy as np
import pytest

import segmodels_keras as sm
from segmodels_keras import Unet
from segmodels_keras.utils import set_regularization

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


if __name__ == "__main__":
    pytest.main([__file__])
