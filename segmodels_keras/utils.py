"""Utility functions for segmentation models"""

from pathlib import Path

from keras import models


def save_model_weights_notop(
    model: models.Model, decoder: str, path: str | Path, overwrite: bool = True
):
    """Save model weights without top (without segmentation head).

    The weights saved like this can be used to preload a segmentation model for
    fine-tuning by passing the path to these weights to ``weights_notop`` argument of
    the model constructor, e.g. ``Unet(weights_notop="path/to/weights.h5")``.

    Args:
        model (``keras.models.Model``): instance of keras model
        decoder: type of the decoder part of the model. Should be one of ``fpn``,
            ``linknet``, ``unet``, ``pspnet``.
        path (str | Path): path to save model weights
        overwrite (bool): whether to overwrite existing file at ``path``.
            Defaults to ``True``.

    """
    decoder_top_layers = {"fpn": 2, "linknet": 2, "unet": 2, "pspnet": 3}
    if decoder not in decoder_top_layers:
        raise ValueError(
            f"Decoder should be one of {decoder_top_layers.keys()}, got {decoder}"
        )

    # Add 1 to the slice to actually remove the number of top layers specified
    nb_top_layers = decoder_top_layers[decoder] + 1
    model_notop = models.Model(model.input, model.layers[-nb_top_layers].output)
    model_notop.save_weights(str(path), overwrite=overwrite)


def set_trainable(model, recompile=True, **kwargs):  # noqa: ARG001
    """Set all layers of model trainable and recompile it

    Note:
        Model is recompiled using same optimizer, loss and metrics::

            model.compile(
                model.optimizer,
                loss=model.loss,
                metrics=model.metrics,
                loss_weights=model.loss_weights,
                sample_weight_mode=model.sample_weight_mode,
                weighted_metrics=model.weighted_metrics,
            )

    Args:
        model (``keras.models.Model``): instance of keras model

    """
    for layer in model.layers:
        layer.trainable = True

    if recompile:
        model.compile(
            model.optimizer,
            loss=model.loss,
            metrics=model.metrics,
            loss_weights=model.loss_weights,
            sample_weight_mode=model.sample_weight_mode,
            weighted_metrics=model.weighted_metrics,
        )


def set_regularization(
    model,
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    beta_regularizer=None,
    gamma_regularizer=None,
    **kwargs,  # noqa: ARG001
):
    """Set regularizers to all layers

    Note:
       Returned model's config is updated correctly

    Args:
        model (``keras.models.Model``): instance of keras model
        kernel_regularizer(``regularizer`): regularizer of kernels
        bias_regularizer(``regularizer``): regularizer of bias
        activity_regularizer(``regularizer``): regularizer of activity
        gamma_regularizer(``regularizer``): regularizer of gamma of BatchNormalization
        beta_regularizer(``regularizer``): regularizer of beta of BatchNormalization

    Return:
        out (``Model``): config updated model
    """
    for layer in model.layers:
        # set kernel_regularizer
        if kernel_regularizer is not None and hasattr(layer, "kernel_regularizer"):
            layer.kernel_regularizer = kernel_regularizer
        # set bias_regularizer
        if bias_regularizer is not None and hasattr(layer, "bias_regularizer"):
            layer.bias_regularizer = bias_regularizer
        # set activity_regularizer
        if activity_regularizer is not None and hasattr(layer, "activity_regularizer"):
            layer.activity_regularizer = activity_regularizer

        # set beta and gamma of BN layer
        if beta_regularizer is not None and hasattr(layer, "beta_regularizer"):
            layer.beta_regularizer = beta_regularizer

        if gamma_regularizer is not None and hasattr(layer, "gamma_regularizer"):
            layer.gamma_regularizer = gamma_regularizer

    out = models.model_from_json(model.to_json())
    out.set_weights(model.get_weights())

    return out
