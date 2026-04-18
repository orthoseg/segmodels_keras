"""ResNet18 and ResNet34 Keras models.

Complements keras.applications (which provides ResNet50+) with BasicBlock-based
ResNet18 and ResNet34. The implementation follows the keras.applications structure
closely so that the two can be maintained together:

  - ``ResNet(stack_fn, ...)`` is the generic builder, matching the signature of
    ``keras.applications.resnet.ResNet``.
  - ``residual_basicblock(x, filters, ..., name)`` is the BasicBlock equivalent of
    ``keras.applications.resnet.residual_block_v1``.
  - ``stack_residual_basicblocks(x, filters, blocks, ..., name)`` groups blocks into a
    stage, equivalent to ``keras.applications.resnet.stack_residual_blocks_v1``.

Layer naming follows the same ``conv{N}_block{M}_*`` convention used by
keras.applications, so models built here share the same weight-loading patterns.

Note that the resulting resnet18 and resnet34 are not exactly the same as the version
in https://github.com/qubvel/classification_models and hence is also different than
the segmentation models in https://github.com/qubvel/segmentation_models.
There are some differences in the bottom and top layers of the resnet, e.g. because
the classification_models version applies also post-activation after the last residual
block.

This implementation is designed to be compatible with torchvision ResNet18/34 pretrained
weights. Because the resnet implementation in keras.applications.resnet.ResNet (which
only supports ResNet50+) already uses the same bottom and top layers this implementation
is based on it, but in addition the following tweaks have been applied:
- BatchNormalization uses ``momentum=0.9`` and ``epsilon=1e-5`` (torchvision defaults)
  instead of the keras.applications defaults of ``momentum=0.99`` and
  ``epsilon=1.001e-5``.
  Note that momentum is only used during training. In addition, the value in pytorch
  is inverted compared to the tf.keras implementation, so the default in pytorch
  is actually ``momentum=0.1``, but after inversion we use ``momentum=0.9`` here.
- The single bottom convolutional layer gets ``use_bias=False`` (like all convolutional
  layers in the model just like the version in torchvision and in segmentation_models.
  In the keras.applications ResNet(50+) implementation ``use_bias=True`` is used for the
  single bottom convolutional layer.
"""

import os

from keras import backend, layers, models
from keras.applications import imagenet_utils
from keras.utils import get_source_inputs

from .resnet_common import _obtain_input_shape


def ResNet(
    stack_fn,
    preact,
    use_bias,
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
    both can be used interchangeably.

    Args:
        stack_fn: A function that returns output tensor for the
            stacked residual blocks.
        preact: Whether to use pre-activation (BN before conv). `True` for ResNetV2,
            `False` (post-activation) for ResNet (including ResNet18/34) and ResNeXt.
        use_bias: Whether to use bias for the single bottom convolutional layer or not.
            `True` for ResNet and ResNetV2 in the standard keras.applications
            implementation, `False` for ResNeXt. However, for torchvision compatibility
            we will use False here for ResNet18/34.
        include_top: Whether to include the fully-connected layer at the top of the
            network (=classification head). Defaults to ``True``.
        weights: Path to a weights file to load, or ``None``. Defaults to
            ``None``.
        input_tensor: Optional Keras tensor to use as the model input.
            (i.e. output of `layers.Input()`)
        input_shape: Optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `"channels_first"` data format). It
            should have exactly 3 inputs channels.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`,
            and if no `weights` argument is specified.
            Defaults to ``1000``.
        classifier_activation: A `str` or callable. The activation
            function to use on the "top" layer. Ignored unless
            `include_top=True`. Set `classifier_activation=None` to
            return the logits of the "top" layer. When loading
            pretrained weights, `classifier_activation` can only be
            `None` or `"softmax"`. Defaults to ``"softmax"``.
        name: The name of the model (string). Defaults to ``"resnet"``.

    Returns:
        A ``keras.Model`` instance.
    """

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if backend.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name="conv1_conv")(x)

    if not preact:
        # The batchnormalization parameters have been changed to be compatible with the
        # torchvision ResNet18/34 pretrained weights.
        x = layers.BatchNormalization(
            axis=bn_axis, momentum=0.1, epsilon=1e-5, name="conv1_bn"
        )(x)
        x = layers.Activation("relu", name="conv1_relu")(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    x = stack_fn(x)

    if preact:
        # The batchnormalization parameters have been changed to be compatible with the
        # torchvision ResNet18/34 pretrained weights.
        x = layers.BatchNormalization(
            axis=bn_axis, momentum=0.1, epsilon=1e-5, name="post_bn"
        )(x)
        x = layers.Activation("relu", name="post_relu")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

        # Validate activation for the classifier layer
        imagenet_utils.validate_activation(classifier_activation, weights)

        x = layers.Dense(classes, activation=classifier_activation, name="predictions")(
            x
        )
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name=name)

    if weights is not None and os.path.exists(weights):
        model.load_weights(weights)

    return model


def residual_basicblock(
    x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None
):
    """A BasicBlock residual block for ResNet18/34.

    Similar to ``keras.applications.resnet.residual_block_v1`` that is used for
    ResNet50+, but with following differences for resnet 18/34:
    - two 3x3 convolutions instead of the 1x1 / 3x3 / 1x1 bottleneck.
    - ``use_bias=False`` instead of ``True``

    Following changes have been applied to be compatible with torchvision ResNet18/34
    pretrained weights:
    - BN momentum is set to 0.1 instead of 0.99
    - BN epsilon is set to 1e-5 instead of 1.001e-5

    Args:
        x: Input tensor.
        filters: No of filters for both 3x3 convolutions.
        kernel_size: Kernel size of the 3x3 convolutions. Defaults to ``3``.
        stride: Stride of the first convolution. Defaults to ``1``.
        conv_shortcut: Use a strided 1x1 conv on the shortcut path if ``True``,
            otherwise use an identity shortcut. Defaults to ``True``.
        name: Block name prefix, e.g. ``"conv3_block1"``.

    Returns:
        Output tensor for the block.
    """
    if backend.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, use_bias=False, name=f"{name}_0_conv"
        )(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, momentum=0.1, epsilon=1e-5, name=f"{name}_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=f"{name}_1_pad")(x)
    x = layers.Conv2D(
        filters, kernel_size, strides=stride, use_bias=False, name=f"{name}_1_conv"
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=0.1, epsilon=1e-5, name=f"{name}_1_bn"
    )(x)
    x = layers.Activation("relu", name=f"{name}_1_relu")(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=f"{name}_2_pad")(x)
    x = layers.Conv2D(filters, kernel_size, use_bias=False, name=f"{name}_2_conv")(x)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=0.1, epsilon=1e-5, name=f"{name}_2_bn"
    )(x)

    x = layers.Add(name=f"{name}_add")([shortcut, x])
    x = layers.Activation("relu", name=f"{name}_out")(x)
    return x


def stack_residual_basicblocks(x, filters, blocks, stride1=2, name=None):
    """A set of stacked BasicBlock residual blocks forming one ResNet stage.

    Similar to ``keras.applications.resnet.stack_residual_blocks_v1``.

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
    x = residual_basicblock(
        x, filters, stride=stride1, conv_shortcut=stride1 > 1, name=f"{name}_block1"
    )
    for i in range(2, blocks + 1):
        x = residual_basicblock(
            x, filters, conv_shortcut=False, name=f"{name}_block{i}"
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
    name="resnet18",
    use_bias=False,
):
    """Instantiates the ResNet18 architecture with torchvision-compatible tweaks.

    Remark: torchvision uses bias=False, keras.applications ResNet uses bias=True, but
    for torchvision compatibility we will use False here.
    """

    def stack_fn(x):
        x = stack_residual_basicblocks(x, 64, 2, stride1=1, name="conv2")
        x = stack_residual_basicblocks(x, 128, 2, stride1=2, name="conv3")
        x = stack_residual_basicblocks(x, 256, 2, stride1=2, name="conv4")
        return stack_residual_basicblocks(x, 512, 2, stride1=2, name="conv5")

    return ResNet(
        stack_fn,
        preact=False,  # torchvision ResNet18/34 is post-activation (Conv -> BN -> ReLU)
        use_bias=use_bias,
        name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def ResNet34(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="resnet34",
    use_bias=False,
):
    """Instantiates the ResNet34 architecture with torchvision-compatible tweaks.

    Remark: torchvision uses bias=False, keras.applications ResNet uses bias=True, but
    for torchvision compatibility we will use False here.
    """

    def stack_fn(x):
        x = stack_residual_basicblocks(x, 64, 3, stride1=1, name="conv2")
        x = stack_residual_basicblocks(x, 128, 4, stride1=2, name="conv3")
        x = stack_residual_basicblocks(x, 256, 6, stride1=2, name="conv4")
        return stack_residual_basicblocks(x, 512, 3, stride1=2, name="conv5")

    return ResNet(
        stack_fn,
        preact=False,  # torchvision ResNet18/34 is post-activation (Conv -> BN -> ReLU)
        use_bias=use_bias,
        name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def preprocess_input(x, **kwargs):
    """Torchvision-compatible ImageNet normalization (input in [0, 1])."""
    import numpy as np

    # FLAIR-compatible normalization (input in [0, 255]).
    mean = np.array([105.08, 110.87, 101.82, 106.38, 53.26], dtype="float32")
    std = np.array([52.17, 45.38, 44, 39.69, 79.3], dtype="float32")

    # Torchvision-compatible ImageNet normalization (input in [0, 1]).
    # mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    # std = np.array([0.229, 0.224, 0.225], dtype="float32")
    return (x - mean) / std
