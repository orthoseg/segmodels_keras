import keras.applications as ka

from . import inception_resnet_v2 as irv2
from . import inception_v3 as iv3


class BackbonesFactory:
    """Dict with all supported backbones.

    Each backbone is represented as a tuple of 3 elements:
    1. Backbone class
    2. Preprocessing function
    3. List of layers to take features from backbone in the following order:
       (x16, x8, x4, x2, x1) - `x4` mean that features has 4 times less spatial
       resolution (Height x Width) than input image.
    """

    _models = {  # noqa: RUF012
        # ResNets
        "resnet50": (
            ka.ResNet50,
            ka.resnet.preprocess_input,
            (
                "conv4_block6_out",
                "conv3_block4_out",
                "conv2_block3_out",
                "conv1_relu",
            ),
        ),
        "resnet101": (
            ka.ResNet101,
            ka.resnet.preprocess_input,
            (
                "conv4_block23_out",
                "conv3_block4_out",
                "conv2_block3_out",
                "conv1_relu",
            ),
        ),
        "resnet152": (
            ka.ResNet152,
            ka.resnet.preprocess_input,
            (
                "conv4_block36_out",
                "conv3_block8_out",
                "conv2_block3_out",
                "conv1_relu",
            ),
        ),
        # VGG
        "vgg16": (
            ka.vgg16.VGG16,
            ka.vgg16.preprocess_input,
            (
                "block5_conv3",
                "block4_conv3",
                "block3_conv3",
                "block2_conv2",
                "block1_conv2",
            ),
        ),
        "vgg19": (
            ka.vgg19.VGG19,
            ka.vgg19.preprocess_input,
            (
                "block5_conv4",
                "block4_conv4",
                "block3_conv4",
                "block2_conv2",
                "block1_conv2",
            ),
        ),
        # DenseNet
        "densenet121": (
            ka.densenet.DenseNet121,
            ka.densenet.preprocess_input,
            # (311, 139, 51, 4),
            ("pool4_conv", "pool3_conv", "pool2_conv", "conv1/relu"),
        ),
        "densenet169": (
            ka.densenet.DenseNet169,
            ka.densenet.preprocess_input,
            # (367, 139, 51, 4),
            ("pool4_conv", "pool3_conv", "pool2_conv", "conv1/relu"),
        ),
        "densenet201": (
            ka.densenet.DenseNet201,
            ka.densenet.preprocess_input,
            # (479, 139, 51, 4),
            ("pool4_conv", "pool3_conv", "pool2_conv", "conv1/relu"),
        ),
        # Inception
        "inceptionresnetv2": (
            irv2.InceptionResNetV2,
            irv2.preprocess_input,
            (594, 260, 16, 9),
        ),
        "inceptionv3": (iv3.InceptionV3, iv3.preprocess_input, (228, 86, 16, 9)),
        # MobileNet
        "mobilenet": (
            ka.mobilenet.MobileNet,
            ka.mobilenet.preprocess_input,
            (
                "conv_pw_11_relu",
                "conv_pw_5_relu",
                "conv_pw_3_relu",
                "conv_pw_1_relu",
            ),
        ),
        "mobilenetv2": (
            ka.mobilenet_v2.MobileNetV2,
            ka.mobilenet_v2.preprocess_input,
            (
                "block_13_expand_relu",
                "block_6_expand_relu",
                "block_3_expand_relu",
                "block_1_expand_relu",
            ),
        ),
        "efficientnetb0": [
            ka.EfficientNetB0,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",
                "block4a_expand_activation",
                "block3a_expand_activation",
                "block2a_expand_activation",
            ),
        ],
        "efficientnetb1": [
            ka.EfficientNetB1,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",
                "block4a_expand_activation",
                "block3a_expand_activation",
                "block2a_expand_activation",
            ),
        ],
        "efficientnetb2": [
            ka.EfficientNetB2,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",
                "block4a_expand_activation",
                "block3a_expand_activation",
                "block2a_expand_activation",
            ),
        ],
        "efficientnetb3": [
            ka.EfficientNetB3,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",
                "block4a_expand_activation",
                "block3a_expand_activation",
                "block2a_expand_activation",
            ),
        ],
        "efficientnetb4": [
            ka.EfficientNetB4,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",
                "block4a_expand_activation",
                "block3a_expand_activation",
                "block2a_expand_activation",
            ),
        ],
        "efficientnetb5": [
            ka.EfficientNetB5,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",
                "block4a_expand_activation",
                "block3a_expand_activation",
                "block2a_expand_activation",
            ),
        ],
        "efficientnetb6": [
            ka.EfficientNetB6,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",
                "block4a_expand_activation",
                "block3a_expand_activation",
                "block2a_expand_activation",
            ),
        ],
        "efficientnetb7": [
            ka.EfficientNetB7,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",
                "block4a_expand_activation",
                "block3a_expand_activation",
                "block2a_expand_activation",
            ),
        ],
    }

    @property
    def models(self):
        return self._models

    def models_names(self):
        return list(self.models.keys())

    def get_backbone(self, name, *args, **kwargs):
        backbone = self._models.get(name)
        if backbone is None:
            raise ValueError(f"Backbone with name '{name}' is not supported.")

        model_fn, _, _ = backbone
        model = model_fn(*args, **kwargs)
        return model

    def get_feature_layers(self, name, n=5):
        return self._models.get(name)[2][:n]

    def get_preprocessing(self, name):
        return self._models.get(name)[1]


Backbones = BackbonesFactory()
