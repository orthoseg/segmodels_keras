"""Model factory for creating segmentation models with common parameters."""

from typing import Literal

from . import fpn, linknet, pspnet, unet

ModelNames = Literal["unet", "linknet", "pspnet", "fpn"]


def get_model(
    model_name: ModelNames,
    backbone_name: str = "vgg16",
    input_shape: tuple[int | None, int | None, int] = (None, None, 3),
    classes: int = 1,
    activation: str = "sigmoid",
    weights: str | None = None,
    weights_notop: str | None = None,
    freeze_notop: bool = False,
    encoder_weights: str | None = "imagenet",
    encoder_freeze: bool = False,
    **kwargs,
):
    """Create a segmentation model with common constructor parameters.

    Args:
        model_name: Name of the model to create. One of 'unet', 'linknet', 'pspnet',
            'fpn'.
        backbone_name: Name of the backbone model. Default is 'vgg16'.
        input_shape: Shape of input data (H, W, C). Default is (None, None, 3).
        classes: Number of output classes. Default is 1.
        activation: Activation function for the last layer. Default is 'sigmoid'.
        weights: Path to model weights to be loaded. Default is None.
        weights_notop: Path to model weights without top (segmentation head) to be
            loaded. Default is None.
        freeze_notop: If True, set all layers except the top as non-trainable.
            Default is False.
        encoder_weights: One of None (random initialization) or 'imagenet'
            (pre-training on ImageNet). Default is 'imagenet'.
        encoder_freeze: If True, set all encoder layers as non-trainable.
            Default is False.
        **kwargs: Additional model-specific parameters. For example:
            - For Unet/Linknet: encoder_features, decoder_block_type, decoder_filters,
              decoder_use_batchnorm
            - For PSPNet: downsample_factor, psp_conv_filters, psp_pooling_type,
              psp_use_batchnorm, psp_dropout
            - For FPN: encoder_features, pyramid_block_filters, pyramid_use_batchnorm,
              pyramid_aggregation, pyramid_dropout

    Returns:
        A compiled Keras segmentation model.

    Raises:
        ValueError: If model_name is not recognized.

    Example:
        >>> model = get_model(
        ...     model_name="unet",
        ...     backbone_name="resnet50",
        ...     classes=3,
        ...     activation="softmax"
        ... )
        >>> model_psp = get_model(
        ...     model_name="pspnet",
        ...     backbone_name="vgg16",
        ...     input_shape=(384, 384, 3),
        ...     classes=21,
        ...     activation="softmax"
        ... )
    """
    model_name = model_name.lower()  # type: ignore[assignment]

    common_params = {
        "backbone_name": backbone_name,
        "input_shape": input_shape,
        "classes": classes,
        "activation": activation,
        "weights": weights,
        "weights_notop": weights_notop,
        "freeze_notop": freeze_notop,
        "encoder_weights": encoder_weights,
        "encoder_freeze": encoder_freeze,
    }

    if model_name == "unet":
        return unet.Unet(**common_params, **kwargs)
    elif model_name == "linknet":
        return linknet.Linknet(**common_params, **kwargs)
    elif model_name == "pspnet":
        return pspnet.PSPNet(**common_params, **kwargs)
    elif model_name == "fpn":
        return fpn.FPN(**common_params, **kwargs)
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Supported models are: {', '.join(ModelNames.__args__)}"
        )
