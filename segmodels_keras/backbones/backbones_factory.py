import keras.applications as ka

from . import inception_resnet_v2 as irv2
from . import inception_v3 as iv3
from . import resnet


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
        # ResNets < 50 layers are NOT available via keras.applications, so use
        # implementation from classification_models.
        "resnet18": (
            resnet.ResNet18,
            resnet.preprocess_input,
            ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
        ),
        "resnet34": (
            resnet.ResNet34,
            resnet.preprocess_input,
            ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
        ),
        # ResNets > 50 layers are available via keras.applications, so use those.
        # Skip layers (inverted) from https://github.com/yingkaisha/keras-unet-collection
        "resnet50": (
            ka.ResNet50,
            ka.resnet.preprocess_input,
            ("conv4_block6_out", "conv3_block4_out", "conv2_block3_out", "conv1_relu"),
        ),
        "resnet101": (
            ka.ResNet101,
            ka.resnet.preprocess_input,
            ("conv4_block23_out", "conv3_block4_out", "conv2_block3_out", "conv1_relu"),
        ),
        "resnet152": (
            ka.ResNet152,
            ka.resnet.preprocess_input,
            ("conv4_block36_out", "conv3_block8_out", "conv2_block3_out", "conv1_relu"),
        ),
        # ResNetV2
        # Skip layers (inverted) from https://github.com/yingkaisha/keras-unet-collection
        "resnet50v2": (
            ka.ResNet50V2,
            ka.resnet_v2.preprocess_input,
            (
                "conv4_block6_1_relu",
                "conv3_block4_1_relu",
                "conv2_block3_1_relu",
                "conv1_conv",
            ),
        ),
        "resnet101v2": (
            ka.ResNet101V2,
            ka.resnet_v2.preprocess_input,
            (
                "conv4_block23_1_relu",
                "conv3_block4_1_relu",
                "conv2_block3_1_relu",
                "conv1_conv",
            ),
        ),
        "resnet152v2": (
            ka.ResNet152V2,
            ka.resnet_v2.preprocess_input,
            (
                "conv4_block36_1_relu",
                "conv3_block8_1_relu",
                "conv2_block3_1_relu",
                "conv1_conv",
            ),
        ),
        # VGG
        # Skip layers from segmentation_models
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
        # Skip layers (inverted) from https://github.com/yingkaisha/keras-unet-collection
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
        # Skip layers from segmentation_models
        "inceptionresnetv2": (
            irv2.InceptionResNetV2,
            irv2.preprocess_input,
            # Use the layer indexes instead of names because otherwise loading weights
            # saved with keras 2 cannot be loaded with keras 3.
            # (
            #     "activation_161",
            #     "activation_74",
            #     "activation_3",
            #     "activation",
            #     # "input_1",
            # ),
            (594, 260, 16, 9),
        ),
        "inceptionv3": (iv3.InceptionV3, iv3.preprocess_input, (228, 86, 16, 9)),
        # MobileNet
        # Skip layers from segmentation_models
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
        # EfficientNet
        # Skip layers from segmentation_models
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
        # EfficientNetV2
        # Skip layers from https://github.com/chinefed/segmentation_models_fork
        "efficientnetv2m": (
            ka.EfficientNetV2M,
            ka.efficientnet_v2.preprocess_input,
            (
                "block6a_expand_conv",
                "block4a_expand_conv",
                "block2e_add",
                "block1c_add",
            ),
        ),
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

    def get_custom_objects(self, name):
        if name == "inceptionresnetv2":
            return irv2.get_custom_objects()
        return {}


Backbones = BackbonesFactory()
