"""ResNet18 and ResNet34 Keras models.

Complements keras.applications (which provides ResNet50+) with BasicBlock-based
ResNet18 and ResNet34. The implementation follows the keras.applications structure
closely so that the two can be maintained together:

  - ``ResNet(stack_fn, ...)`` is the generic builder, matching the signature of
    ``keras.applications.resnet.ResNet``.
  - ``block_for_resnet(x, filters, ..., name)`` is the BasicBlock equivalent of
    ``keras.applications.resnet.residual_block_v1``.
  - ``stack_residual_blocks(x, filters, blocks, ..., name)`` groups blocks into a
    stage, equivalent to ``keras.applications.resnet.stack_residual_blocks_v1``.

Layer naming follows the same ``conv{N}_block{M}_*`` convention used by
keras.applications, so models built here share the same weight-loading patterns.

BN uses ``momentum=0.1`` and ``epsilon=1e-5`` (torchvision defaults) instead of
the keras.applications value of ``epsilon=1.001e-5``, to allow direct weight
transfer from torchvision pretrained models.
"""

import os

from keras import backend, layers, models


def ResNet(
    stack_fn,
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="resnet",
):
    """Generic ResNet builder for BasicBlock variants (ResNet18/34).

    Follows the same signature as ``keras.applications.resnet.ResNet`` so that
    both can be used interchangeably.  The ``preact`` and ``use_bias`` arguments
    of the keras.applications version are omitted because BasicBlock ResNets are
    always post-activation and always use ``use_bias=False`` on conv layers.

    Args:
        stack_fn: A callable that takes the post-stem tensor and returns the
            output of the final residual stage.
        include_top: Whether to include the classification head. Defaults to
            ``True``.
        weights: Path to a weights file to load, or ``None``. Defaults to
            ``None``.
        input_tensor: Optional Keras tensor to use as the model input.
        input_shape: Optional input shape tuple (used when ``input_tensor`` is
            ``None``).
        pooling: Pooling mode when ``include_top=False``: ``"avg"``, ``"max"``,
            or ``None``. Defaults to ``None``.
        classes: Number of output classes for the classification head. Defaults
            to ``1000``.
        classifier_activation: Activation for the classification head. Defaults
            to ``"softmax"``.
        name: Model name. Defaults to ``"resnet"``.

    Returns:
        A ``keras.Model`` instance.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name="conv1_conv")(x)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=0.1, epsilon=1e-5, name="conv1_bn"
    )(x)
    x = layers.Activation("relu", name="conv1_relu")(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    x = stack_fn(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(classes, activation=classifier_activation, name="predictions")(
            x
        )
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    inputs = img_input if input_tensor is None else layers.Input(tensor=input_tensor)
    model = models.Model(inputs, x, name=name)

    if weights is not None and os.path.exists(weights):
        model.load_weights(weights)

    return model


def block_for_resnet(
    x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None
):
    """A BasicBlock residual block for ResNet18/34.

    Equivalent to ``keras.applications.resnet.residual_block_v1`` but with two
    3x3 convolutions instead of the 1x1 / 3x3 / 1x1 bottleneck used by ResNet50+.

    Args:
        x: Input tensor.
        filters: Number of filters for both 3x3 convolutions.
        kernel_size: Kernel size of the 3x3 convolutions. Defaults to ``3``.
        stride: Stride of the first convolution. Defaults to ``1``.
        conv_shortcut: Use a strided 1x1 conv on the shortcut path if ``True``,
            otherwise use an identity shortcut. Defaults to ``True``.
        name: Block name prefix, e.g. ``"conv3_block1"``.

    Returns:
        Output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, use_bias=False, name=name + "_0_conv"
        )(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, momentum=0.1, epsilon=1e-5, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + "_1_pad")(x)
    x = layers.Conv2D(
        filters, kernel_size, strides=stride, use_bias=False, name=name + "_1_conv"
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=0.1, epsilon=1e-5, name=name + "_1_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + "_2_pad")(x)
    x = layers.Conv2D(filters, kernel_size, use_bias=False, name=name + "_2_conv")(x)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=0.1, epsilon=1e-5, name=name + "_2_bn"
    )(x)

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation("relu", name=name + "_out")(x)
    return x


def stack_residual_blocks(x, filters, blocks, stride1=2, name=None):
    """A set of stacked BasicBlock residual blocks forming one ResNet stage.

    Equivalent to ``keras.applications.resnet.stack_residual_blocks_v1``.

    Args:
        x: Input tensor.
        filters: Number of filters for all blocks in this stage.
        blocks: Number of BasicBlocks in the stage.
        stride1: Stride of the first block (use ``2`` to downsample spatially,
            ``1`` for the first stage where spatial size is unchanged). Defaults
            to ``2``.
        name: Stage name prefix, e.g. ``"conv3"``.

    Returns:
        Output tensor for the stacked blocks.
    """
    x = block_for_resnet(
        x, filters, stride=stride1, conv_shortcut=stride1 > 1, name=name + "_block1"
    )
    for i in range(2, blocks + 1):
        x = block_for_resnet(
            x, filters, conv_shortcut=False, name=name + "_block" + str(i)
        )
    return x


def ResNet18(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ResNet18 architecture with torchvision-compatible BasicBlocks."""

    def stack_fn(x):
        x = stack_residual_blocks(x, 64, 2, stride1=1, name="conv2")
        x = stack_residual_blocks(x, 128, 2, stride1=2, name="conv3")
        x = stack_residual_blocks(x, 256, 2, stride1=2, name="conv4")
        return stack_residual_blocks(x, 512, 2, stride1=2, name="conv5")

    return ResNet(
        stack_fn,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        name="resnet18",
    )


def ResNet34(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ResNet34 architecture with torchvision-compatible BasicBlocks."""

    def stack_fn(x):
        x = stack_residual_blocks(x, 64, 3, stride1=1, name="conv2")
        x = stack_residual_blocks(x, 128, 4, stride1=2, name="conv3")
        x = stack_residual_blocks(x, 256, 6, stride1=2, name="conv4")
        return stack_residual_blocks(x, 512, 3, stride1=2, name="conv5")

    return ResNet(
        stack_fn,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        name="resnet34",
    )


def preprocess_input(x, **kwargs):
    """Torchvision-compatible ImageNet normalization (input in [0, 1])."""
    import numpy as np

    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    return (x - mean) / std
