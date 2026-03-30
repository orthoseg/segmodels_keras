from keras import layers


def freeze_model(model, **kwargs):  # noqa: ARG001
    """Set all layers non trainable, excluding BatchNormalization layers"""
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def filter_keras_submodules(kwargs):
    """Selects only arguments that define keras_application submodules."""
    submodule_keys = kwargs.keys() & {"backend", "layers", "models", "utils"}
    return {key: kwargs[key] for key in submodule_keys}
