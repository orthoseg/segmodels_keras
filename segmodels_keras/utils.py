"""Utility functions for segmentation models."""

from collections import defaultdict
from pathlib import Path
from typing import Any

from keras import models


def load_weights(model: Any, filepath: str | Path) -> None:
    """Load weights from an HDF5 file into a Keras model.

    This is an enhanced wrapper around ``model.load_weights(filepath)`` that provides
    compatibility with both Keras 3 and legacy Keras 2 HDF5 weight files.

    TensorFlow/Keras 2.10 and 2.11 can only read the legacy HDF5 format
    (``layer_names`` attribute at the root).  Keras 3 switched to a newer
    layout where weights are stored under ``layers/<name>/vars/<index>``.

    When a file saved with Keras 3 is opened by the TF 2.10/2.11 loader the
    error "found 0 saved layers" is raised because the legacy reader finds no
    ``layer_names`` attribute.

    When a file saved with Keras 2 is opened by Keras 3, the error about
    "expected X variables, but received 0 variables" is raised because Keras 3
    cannot read the legacy format directly.

    This wrapper calls ``model.load_weights`` first. If that raises a ValueError:
    - If it's a Keras 3 weights file loaded in Keras 2, it uses a custom reader
      that matches each model layer to its saved counterpart by weight-shape
      signature and file creation order.
    - If it's a Keras 2 weights file loaded in Keras 3, it delegates to Keras'
      legacy HDF5 format loader.

    Args:
        model: Keras model whose weights should be restored.
        filepath (str | Path): Path to a ``.h5`` or ``.hdf5`` weights file.
    """
    try:
        model.load_weights(filepath)
    except ValueError as ex:
        filepath = str(filepath)
        message = str(ex)
        is_hdf5 = Path(filepath).suffix in {".h5", ".hdf5"}

        if not is_hdf5:
            raise

        # Try to handle Keras 3 weights in Keras 2
        if "found 0 saved layers" in message and _is_keras_v3_weights_hdf5(filepath):
            _load_keras_v3_weights_hdf5(model, filepath)
            return

        # Try to handle Keras 2 weights in Keras 3
        if (
            "expected" in message
            and "variables" in message
            and "received 0" in message
            and _is_keras_v2_weights_hdf5(filepath)
        ):
            _load_keras_v2_weights_hdf5(model, filepath)
            return

        # If none of the above, re-raise the original error
        raise


def _is_keras_v3_weights_hdf5(filepath: str | Path) -> bool:
    """Check if the HDF5 file is in Keras 3 format.

    Keras 3 HDF5 weight files have a "layers" group with subgroups for each layer,
    and each layer subgroup contains a "vars" group with datasets for each weight
    variable.
    """
    try:
        import h5py  # noqa: PLC0415
    except ImportError:
        return False

    try:
        with h5py.File(filepath, "r") as handle:
            return "layers" in handle and "vars" in handle
    except OSError:
        return False


def _is_keras_v2_weights_hdf5(filepath: str | Path) -> bool:
    """Check if the HDF5 file is in Keras 2 format.

    Keras 2 HDF5 weight files have a "layer_names" attribute at the root.
    """
    try:
        import h5py  # noqa: PLC0415
    except ImportError:
        return False

    try:
        with h5py.File(filepath, "r") as handle:
            # Keras 2 format has layer_names attribute at the root
            return "layer_names" in handle.attrs
    except OSError:
        return False


def _get_sorted_hdf5_var_keys(vars_group: Any) -> list[int]:
    return sorted(vars_group.keys(), key=int)


def _get_keras_v3_weighted_layers(handle: Any) -> list[dict[str, Any]]:
    import h5py  # noqa: PLC0415

    weighted_layers = []
    for layer_name in handle["layers"].keys():
        layer_group = handle["layers"][layer_name]
        vars_group = layer_group.get("vars")
        if vars_group is None or not vars_group.keys():
            continue

        signature = tuple(
            tuple(vars_group[key].shape)
            for key in _get_sorted_hdf5_var_keys(vars_group)
        )
        weighted_layers.append(
            {
                "name": layer_name,
                "addr": h5py.h5o.get_info(layer_group.id).addr,
                "signature": signature,
                "vars_group": vars_group,
            }
        )

    weighted_layers.sort(key=lambda layer: layer["addr"])
    return weighted_layers


def _load_keras_v3_weights_hdf5(model: Any, filepath: str | Path) -> None:
    import h5py  # noqa: PLC0415

    with h5py.File(filepath, "r") as handle:
        saved_layers = _get_keras_v3_weighted_layers(handle)
        saved_layers_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for saved_layer in saved_layers:
            saved_layers_by_signature[saved_layer["signature"]].append(saved_layer)

        loaded_layers = 0

        for layer in model.layers:
            if not layer.weights:
                continue

            signature = tuple(tuple(variable.shape) for variable in layer.weights)
            if not saved_layers_by_signature[signature]:
                raise ValueError(
                    "Unable to locate matching Keras v3 HDF5 weights for layer "
                    f"'{layer.name}' with shapes {list(signature)}."
                )

            vars_group = saved_layers_by_signature[signature].pop(0)["vars_group"]
            weight_values = [
                vars_group[key][()] for key in _get_sorted_hdf5_var_keys(vars_group)
            ]
            layer.set_weights(weight_values)
            loaded_layers += 1

    if loaded_layers == 0:
        raise ValueError(
            "No matching weighted layers found in Keras v3 HDF5 weights file: "
            f"{filepath}"
        )


def _load_keras_v2_weights_hdf5(model: Any, filepath: str | Path) -> None:
    """Load legacy Keras 2 HDF5 weights into a Keras 3 model.

    Uses Keras' built-in legacy HDF5 loader to restore weights from files
    saved with Keras 2, which store weights under ``layer_names`` groups
    rather than the modern ``layers/<name>/vars`` structure.
    """
    import h5py  # noqa: PLC0415
    from keras.src.legacy.saving import legacy_h5_format  # noqa: PLC0415

    with h5py.File(filepath, "r") as handle:
        if "layer_names" not in handle.attrs and "model_weights" in handle:
            handle = handle["model_weights"]

        if "layer_names" not in handle.attrs:
            raise ValueError(
                f"HDF5 file {filepath} does not appear to be in Keras 2 format "
                "(missing layer_names attribute)"
            )

        legacy_h5_format.load_weights_from_hdf5_group(handle, model)


def save_model_weights_notop(
    model: models.Model, decoder: str, path: str | Path, overwrite: bool = True
) -> None:
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
    decoder_top_layers: dict[str, int] = {
        "fpn": 2,
        "linknet": 2,
        "unet": 2,
        "pspnet": 3,
    }
    if decoder not in decoder_top_layers:
        raise ValueError(
            f"Decoder should be one of {decoder_top_layers.keys()}, got {decoder}"
        )

    # Add 1 to the slice to actually remove the number of top layers specified
    nb_top_layers = decoder_top_layers[decoder] + 1
    model_notop = models.Model(model.input, model.layers[-nb_top_layers].output)
    model_notop.save_weights(str(path), overwrite=overwrite)


def set_trainable(model: models.Model, recompile: bool = True, **kwargs: Any) -> None:  # noqa: ARG001
    """Set all layers of model trainable and recompile it.

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
        model (``keras.models.Model``): instance of keras model.
        recompile: whether to recompile the model after setting trainable.
        **kwargs: additional keyword arguments (unused).

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
    model: models.Model,
    kernel_regularizer: Any = None,
    bias_regularizer: Any = None,
    activity_regularizer: Any = None,
    beta_regularizer: Any = None,
    gamma_regularizer: Any = None,
    **kwargs: dict[str, Any],  # noqa: ARG001
) -> models.Model:
    """Set regularizers to all layers.

    Note:
       Returned model's config is updated correctly

    Args:
        model: instance of keras model.
        kernel_regularizer: regularizer of kernels.
        bias_regularizer: regularizer of bias.
        activity_regularizer: regularizer of activity.
        gamma_regularizer: regularizer of gamma of BatchNormalization.
        beta_regularizer: regularizer of beta of BatchNormalization.
        **kwargs: additional keyword arguments (unused).

    Returns:
        out: config updated model.
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
