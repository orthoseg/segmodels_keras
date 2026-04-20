from collections import defaultdict
from pathlib import Path

import keras
from packaging.version import parse as parse_version

KERAS_GTE_3 = parse_version(keras.__version__) >= parse_version("3.0.0")


def _is_keras_v3_weights_hdf5(filepath):
    try:
        import h5py  # noqa: PLC0415
    except ImportError:
        return False

    try:
        with h5py.File(filepath, "r") as handle:
            return "layers" in handle and "vars" in handle
    except OSError:
        return False


def _get_sorted_hdf5_var_keys(vars_group):
    return sorted(vars_group.keys(), key=int)


def _get_keras_v3_weighted_layers(handle):
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


def _load_keras_v3_weights_hdf5(model, filepath):
    import h5py  # noqa: PLC0415

    with h5py.File(filepath, "r") as handle:
        saved_layers = _get_keras_v3_weighted_layers(handle)
        saved_layers_by_signature = defaultdict(list)
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


def load_weights(model, filepath):
    """Load weights from an HDF5 file into a Keras model.

    TensorFlow/Keras 2.10 and 2.11 can only read the legacy HDF5 format
    (``layer_names`` attribute at the root).  Keras 3 switched to a newer
    layout where weights are stored under ``layers/<name>/vars/<index>``.
    When a file saved with Keras 3 is opened by the TF 2.10/2.11 loader the
    error "found 0 saved layers" is raised because the legacy reader finds no
    ``layer_names`` attribute.

    This wrapper calls ``model.load_weights`` first.  If that raises the
    "found 0 saved layers" ValueError *and* the file is detected as a Keras 3
    weights HDF5 file, it falls back to a manual reader that matches each
    model layer to its saved counterpart by weight-shape signature and file
    creation order.

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
        if (
            not is_hdf5
            or "found 0 saved layers" not in message
            or not _is_keras_v3_weights_hdf5(filepath)
        ):
            raise

        _load_keras_v3_weights_hdf5(model, filepath)
