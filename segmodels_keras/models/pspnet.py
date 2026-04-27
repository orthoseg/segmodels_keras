from typing import Any

from keras import backend, layers, models
from keras import utils as keras_utils

from ..backbones.backbones_factory import Backbones
from ..utils import load_weights
from ._common_blocks import Conv2dBn
from ._utils import freeze_model

# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------


def get_submodules() -> dict[str, Any]:
    return {
        "backend": backend,
        "models": models,
        "layers": layers,
        "utils": keras_utils,
    }


def check_input_shape(
    input_shape: tuple[int, int, int] | tuple[int | None, int | None, int],
    factor: int,
) -> None:
    if input_shape is None:
        raise ValueError("Input shape should be a tuple of 3 integers, not None!")

    h, w = (
        input_shape[:2]
        if backend.image_data_format() == "channels_last"
        else input_shape[1:]
    )
    if h is None or w is None:
        raise ValueError(
            "Input shape should define spatial dimensions for PSPNet, "
            f"got {input_shape}!"
        )

    min_size = factor * 6

    is_wrong_shape = (
        h % min_size != 0 or w % min_size != 0 or h < min_size or w < min_size
    )

    if is_wrong_shape:
        raise ValueError(
            f"Wrong shape {input_shape}, input H and W should "
            f"be divisible by `{min_size}`"
        )


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------


def Conv1x1BnReLU(
    filters: int,
    use_batchnorm: bool,
    name: str | None = None,
) -> Any:
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=1,
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs,
        )(input_tensor)

    return wrapper


def SpatialContextBlock(
    level: int,
    conv_filters: int = 512,
    pooling_type: str = "avg",
    use_batchnorm: bool = True,
) -> Any:
    if pooling_type not in ("max", "avg"):
        raise ValueError(
            f"Unsupported pooling type - `{pooling_type}`.Use `avg` or `max`."
        )

    Pooling2D = layers.MaxPool2D if pooling_type == "max" else layers.AveragePooling2D

    pooling_name = f"psp_level{level}_pooling"
    conv_block_name = f"psp_level{level}"
    upsampling_name = f"psp_level{level}_upsampling"

    def wrapper(input_tensor):
        # extract input feature maps size (h, and w dimensions)
        input_shape = input_tensor.shape
        spatial_size = (
            input_shape[1:3]
            if backend.image_data_format() == "channels_last"
            else input_shape[2:]
        )

        # Compute the kernel and stride sizes according to how large the final feature
        # map will be when the kernel factor and strides are equal, then we can compute
        # the final feature map factor by simply dividing the current factor by the
        # kernel or stride factor the final feature map sizes are 1x1, 2x2, 3x3, and
        # 6x6.
        pool_size = up_size = [spatial_size[0] // level, spatial_size[1] // level]

        x = Pooling2D(pool_size, strides=pool_size, padding="same", name=pooling_name)(
            input_tensor
        )
        x = Conv1x1BnReLU(conv_filters, use_batchnorm, name=conv_block_name)(x)
        x = layers.UpSampling2D(
            up_size, interpolation="bilinear", name=upsampling_name
        )(x)
        return x

    return wrapper


# ---------------------------------------------------------------------
#  PSP Decoder
# ---------------------------------------------------------------------


def build_psp(
    backbone: models.Model,
    psp_layer_idx: int | str,
    pooling_type: str = "avg",
    conv_filters: int = 512,
    use_batchnorm: bool = True,
    final_upsampling_factor: int = 8,
    classes: int = 21,
    activation: str = "softmax",
    dropout: float | None = None,
    weights_notop: str | None = None,
    freeze_notop: bool = False,
) -> models.Model:
    input_ = backbone.input
    x = (
        backbone.get_layer(name=psp_layer_idx).output
        if isinstance(psp_layer_idx, str)
        else backbone.get_layer(index=psp_layer_idx).output
    )

    # build spatial pyramid
    x1 = SpatialContextBlock(1, conv_filters, pooling_type, use_batchnorm)(x)
    x2 = SpatialContextBlock(2, conv_filters, pooling_type, use_batchnorm)(x)
    x3 = SpatialContextBlock(3, conv_filters, pooling_type, use_batchnorm)(x)
    x6 = SpatialContextBlock(6, conv_filters, pooling_type, use_batchnorm)(x)

    # aggregate spatial pyramid
    concat_axis = 3 if backend.image_data_format() == "channels_last" else 1
    x = layers.Concatenate(axis=concat_axis, name="psp_concat")([x, x1, x2, x3, x6])
    x = Conv1x1BnReLU(conv_filters, use_batchnorm, name="aggregation")(x)

    # model regularization
    if dropout is not None:
        x = layers.SpatialDropout2D(dropout, name="spatial_dropout")(x)

    # load weights without top if path provided
    if weights_notop is not None:
        model_notop = models.Model(input_, x)
        load_weights(model_notop, weights_notop)
        if freeze_notop:
            freeze_model(model_notop)
        x = model_notop.output

    # model head
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="glorot_uniform",
        name="final_conv",
    )(x)

    x = layers.UpSampling2D(
        final_upsampling_factor, name="final_upsampling", interpolation="bilinear"
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    model = models.Model(input_, x)

    return model


# ---------------------------------------------------------------------
#  PSP Model
# ---------------------------------------------------------------------


def PSPNet(
    backbone_name: str = "vgg16",
    input_shape: tuple[int, int, int] = (384, 384, 3),
    classes: int = 21,
    activation: str = "softmax",
    weights: str | None = None,
    weights_notop: str | None = None,
    freeze_notop: bool = False,
    encoder_weights: str | None = "imagenet",
    encoder_freeze: bool = False,
    downsample_factor: int = 8,
    psp_conv_filters: int = 512,
    psp_pooling_type: str = "avg",
    psp_use_batchnorm: bool = True,
    psp_dropout: float | None = None,
    **kwargs,
) -> models.Model:
    """PSPNet_ is a fully convolution neural network for image semantic segmentation

    Args:
        backbone_name: name of classification model used as feature
                extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``.
            ``H`` and ``W`` should be divisible by ``6 * downsample_factor`` and
            **NOT** ``None``!
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
            (e.g. ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights to be loaded.
        weights_notop: optional, path to model weights without top (without segmentation
            head) to be loaded.
        freeze_notop: if ``True``, set all layers of the model except the top as
            non-trainable.
        encoder_weights: one of ``None`` (random initialization), ``imagenet``
            (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as
            non-trainable.
        downsample_factor: one of 4, 8 and 16. Downsampling rate or in other words
            backbone depth to construct PSP module on it.
        psp_conv_filters: number of filters in ``Conv2D`` layer in each PSP block.
        psp_pooling_type: one of 'avg', 'max'. PSP block pooling type
            (maximum or average).
        psp_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D``
            and ``Activation`` layers is used.
        psp_dropout: dropout rate between 0 and 1.

    Returns:
        ``keras.models.Model``: **PSPNet**

    .. _PSPNet:
        https://arxiv.org/pdf/1612.01105.pdf

    """
    # control image input shape
    check_input_shape(input_shape, downsample_factor)

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )

    feature_layers = Backbones.get_feature_layers(backbone_name, n=3)

    if downsample_factor == 16:
        psp_layer_idx = feature_layers[0]
    elif downsample_factor == 8:
        psp_layer_idx = feature_layers[1]
    elif downsample_factor == 4:
        psp_layer_idx = feature_layers[2]
    else:
        raise ValueError(f"Unsupported factor - `{downsample_factor}`, Use 4, 8 or 16.")

    model = build_psp(
        backbone,
        psp_layer_idx,
        pooling_type=psp_pooling_type,
        conv_filters=psp_conv_filters,
        use_batchnorm=psp_use_batchnorm,
        final_upsampling_factor=downsample_factor,
        classes=classes,
        activation=activation,
        dropout=psp_dropout,
        weights_notop=weights_notop,
        freeze_notop=freeze_notop,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        load_weights(model, weights)

    return model
