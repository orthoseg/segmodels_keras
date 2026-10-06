"""Backbones factory module."""

import keras
import keras.applications as ka

from . import inception_resnet_v2 as irv2
from . import inception_v3 as iv3
from . import resnet_18_34


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
        # Feature sizes in comments are HxWxC for a 512x512 input.
        # ResNets < 50 layers are NOT available via keras.applications, so use a
        # torchvision-compatible post-activation BasicBlock implementation.
        # Layer names follow the keras.applications ResNet50+ naming convention.
        "resnet18": (
            resnet_18_34.ResNet18,
            resnet_18_34.preprocess_input,
            # resnet18: layer1=2, layer2=2, layer3=2, layer4=2 blocks
            # backbone output = conv5_block2_out (512ch); skips are the 4 stages before
            (
                "conv4_block2_out",  # 32x32x256
                "conv3_block2_out",  # 64x64x128
                "conv2_block2_out",  # 128x128x64
                "conv1_relu",  # 256x256x64
            ),
        ),
        "resnet34": (
            resnet_18_34.ResNet34,
            resnet_18_34.preprocess_input,
            # resnet34: layer1=3, layer2=4, layer3=6, layer4=3 blocks
            # backbone output = conv5_block3_out (512ch); skips are the 4 stages before
            (
                "conv4_block6_out",  # 32x32x256
                "conv3_block4_out",  # 64x64x128
                "conv2_block3_out",  # 128x128x64
                "conv1_relu",  # 256x256x64
            ),
        ),
        # ResNets > 50 layers are available via keras.applications, so use those.
        # Skip layers (inverted) from https://github.com/yingkaisha/keras-unet-collection
        "resnet50": (
            ka.ResNet50,
            ka.resnet.preprocess_input,
            (
                "conv4_block6_out",  # 32x32x1024
                "conv3_block4_out",  # 64x64x512
                "conv2_block3_out",  # 128x128x256
                "conv1_relu",  # 256x256x64
            ),
        ),
        "resnet101": (
            ka.ResNet101,
            ka.resnet.preprocess_input,
            (
                "conv4_block23_out",  # 32x32x1024
                "conv3_block4_out",  # 64x64x512
                "conv2_block3_out",  # 128x128x256
                "conv1_relu",  # 256x256x64
            ),
        ),
        "resnet152": (
            ka.ResNet152,
            ka.resnet.preprocess_input,
            (
                "conv4_block36_out",  # 32x32x1024
                "conv3_block8_out",  # 64x64x512
                "conv2_block3_out",  # 128x128x256
                "conv1_relu",  # 256x256x64
            ),
        ),
        # ResNetV2
        # Skip layers (inverted) from https://github.com/yingkaisha/keras-unet-collection
        "resnet50v2": (
            ka.ResNet50V2,
            ka.resnet_v2.preprocess_input,
            (
                "conv4_block6_1_relu",  # 32x32x256
                "conv3_block4_1_relu",  # 64x64x128
                "conv2_block3_1_relu",  # 128x128x64
                "conv1_conv",  # 256x256x64
            ),
        ),
        "resnet101v2": (
            ka.ResNet101V2,
            ka.resnet_v2.preprocess_input,
            (
                "conv4_block23_1_relu",  # 32x32x256
                "conv3_block4_1_relu",  # 64x64x128
                "conv2_block3_1_relu",  # 128x128x64
                "conv1_conv",  # 256x256x64
            ),
        ),
        "resnet152v2": (
            ka.ResNet152V2,
            ka.resnet_v2.preprocess_input,
            (
                "conv4_block36_1_relu",  # 32x32x256
                "conv3_block8_1_relu",  # 64x64x128
                "conv2_block3_1_relu",  # 128x128x64
                "conv1_conv",  # 256x256x64
            ),
        ),
        # VGG
        # Skip layers from segmentation_models
        "vgg16": (
            ka.vgg16.VGG16,
            ka.vgg16.preprocess_input,
            (
                "block5_conv3",  # 32x32x512
                "block4_conv3",  # 64x64x512
                "block3_conv3",  # 128x128x256
                "block2_conv2",  # 256x256x128
                "block1_conv2",  # 512x512x64
            ),
        ),
        "vgg19": (
            ka.vgg19.VGG19,
            ka.vgg19.preprocess_input,
            (
                "block5_conv4",  # 32x32x512
                "block4_conv4",  # 64x64x512
                "block3_conv4",  # 128x128x256
                "block2_conv2",  # 256x256x128
                "block1_conv2",  # 512x512x64
            ),
        ),
        # DenseNet
        # Skip layers (inverted) from https://github.com/yingkaisha/keras-unet-collection
        "densenet121": (
            ka.densenet.DenseNet121,
            ka.densenet.preprocess_input,
            (
                "pool4_conv",  # 32x32x512
                "pool3_conv",  # 64x64x256
                "pool2_conv",  # 128x128x128
                "conv1/relu",  # 256x256x64
            ),
        ),
        "densenet169": (
            ka.densenet.DenseNet169,
            ka.densenet.preprocess_input,
            (
                "pool4_conv",  # 32x32x640
                "pool3_conv",  # 64x64x256
                "pool2_conv",  # 128x128x128
                "conv1/relu",  # 256x256x64
            ),
        ),
        "densenet201": (
            ka.densenet.DenseNet201,
            ka.densenet.preprocess_input,
            (
                "pool4_conv",  # 32x32x896
                "pool3_conv",  # 64x64x256
                "pool2_conv",  # 128x128x128
                "conv1/relu",  # 256x256x64
            ),
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
            (
                594,  # 32x32x1088
                260,  # 64x64x320
                16,  # 128x128x192
                9,  # 256x256x64
            ),
        ),
        "inceptionv3": (
            iv3.InceptionV3,
            iv3.preprocess_input,
            (
                228,  # 32x32x768
                86,  # 64x64x288
                16,  # 128x128x192
                9,  # 256x256x64
            ),
        ),
        # MobileNet
        # Skip layers from segmentation_models
        "mobilenet": (
            ka.mobilenet.MobileNet,
            ka.mobilenet.preprocess_input,
            (
                "conv_pw_11_relu",  # 32x32x512
                "conv_pw_5_relu",  # 64x64x256
                "conv_pw_3_relu",  # 128x128x128
                "conv_pw_1_relu",  # 256x256x64
            ),
        ),
        "mobilenetv2": (
            ka.mobilenet_v2.MobileNetV2,
            ka.mobilenet_v2.preprocess_input,
            (
                "block_13_expand_relu",  # 32x32x576
                "block_6_expand_relu",  # 64x64x192
                "block_3_expand_relu",  # 128x128x144
                "block_1_expand_relu",  # 256x256x96
            ),
        ),
        # EfficientNet
        # Skip layers from segmentation_models
        "efficientnetb0": [
            ka.EfficientNetB0,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",  # 32x32x672
                "block4a_expand_activation",  # 64x64x240
                "block3a_expand_activation",  # 128x128x144
                "block2a_expand_activation",  # 256x256x96
            ),
        ],
        "efficientnetb1": [
            ka.EfficientNetB1,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",  # 32x32x672
                "block4a_expand_activation",  # 64x64x240
                "block3a_expand_activation",  # 128x128x144
                "block2a_expand_activation",  # 256x256x96
            ),
        ],
        "efficientnetb2": [
            ka.EfficientNetB2,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",  # 32x32x720
                "block4a_expand_activation",  # 64x64x288
                "block3a_expand_activation",  # 128x128x144
                "block2a_expand_activation",  # 256x256x96
            ),
        ],
        "efficientnetb3": [
            ka.EfficientNetB3,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",  # 32x32x816
                "block4a_expand_activation",  # 64x64x288
                "block3a_expand_activation",  # 128x128x192
                "block2a_expand_activation",  # 256x256x144
            ),
        ],
        "efficientnetb4": [
            ka.EfficientNetB4,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",  # 32x32x960
                "block4a_expand_activation",  # 64x64x336
                "block3a_expand_activation",  # 128x128x192
                "block2a_expand_activation",  # 256x256x144
            ),
        ],
        "efficientnetb5": [
            ka.EfficientNetB5,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",  # 32x32x1056
                "block4a_expand_activation",  # 64x64x384
                "block3a_expand_activation",  # 128x128x240
                "block2a_expand_activation",  # 256x256x144
            ),
        ],
        "efficientnetb6": [
            ka.EfficientNetB6,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",  # 32x32x1200
                "block4a_expand_activation",  # 64x64x432
                "block3a_expand_activation",  # 128x128x240
                "block2a_expand_activation",  # 256x256x192
            ),
        ],
        "efficientnetb7": [
            ka.EfficientNetB7,
            ka.efficientnet.preprocess_input,
            (
                "block6a_expand_activation",  # 32x32x1344
                "block4a_expand_activation",  # 64x64x480
                "block3a_expand_activation",  # 128x128x288
                "block2a_expand_activation",  # 256x256x192
            ),
        ],
        # EfficientNetV2
        # Stage layout differs from EfficientNet-Bx.
        # Skip layers selected to match decoder resolutions:
        # x16 -> x8 -> x4 -> x2
        #
        # Tests with more compact skip connection layers ("block5i_add" (32x32x160),
        # "block3d_add" (64x64x64), "block2d_add" (128x128x48), "stem_activation"
        # (256x256x24)) showed that the decrease in model size was small (30 MB) but
        # the accuracy performance impact was significant (-1.5%).
        "efficientnetv2s": (
            ka.EfficientNetV2S,
            ka.efficientnet_v2.preprocess_input,
            (
                "block6a_expand_conv",  # 32x32x960
                "block4a_expand_conv",  # 64x64x256
                "block2a_expand_activation",  # 128x128x96
                "stem_activation",  # 256x256x24
            ),
        ),
        "efficientnetv2m": (
            ka.EfficientNetV2M,
            ka.efficientnet_v2.preprocess_input,
            (
                "block6a_expand_conv",  # 32x32x1056
                "block4a_expand_conv",  # 64x64x320
                "block2a_expand_activation",  # 128x128x96
                "stem_activation",  # 256x256x24
            ),
        ),
        "efficientnetv2l": (
            ka.EfficientNetV2L,
            ka.efficientnet_v2.preprocess_input,
            (
                "block6a_expand_conv",  # 32x32x1344
                "block4a_expand_conv",  # 64x64x384
                "block2a_expand_activation",  # 128x128x128
                "stem_activation",  # 256x256x32
            ),
        ),
    }

    @property
    def models(self):
        """Get the dictionary of available backbones."""
        return self._models

    def models_names(self):
        """Get the list of available backbone names."""
        return list(self.models.keys())

        """Get a backbone model by name."""

    def get_backbone(self, name, *args, **kwargs) -> keras.Model:
        """Get a backbone model by name."""
        backbone = self._models.get(name)
        if backbone is None:
            raise ValueError(f"Backbone with name '{name}' is not supported.")

        model_fn, _, _ = backbone
        model = model_fn(*args, **kwargs)
        return model

    def get_feature_layers(self, name, n=5):
        """Get the list of skip layers for a backbone by name."""
        return self._models.get(name)[2][:n]

    def get_preprocessing(self, name):
        """Get the preprocessing function for a backbone by name."""
        return self._models.get(name)[1]

    def get_custom_objects(self, name):
        """Get the custom objects for a backbone by name."""
        if name == "inceptionresnetv2":
            return irv2.get_custom_objects()
        return {}


Backbones = BackbonesFactory()
