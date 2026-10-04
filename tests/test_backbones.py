"""Tests on the backbone factory and backbone retrieval."""

import pytest

from segmodels_keras.backbones.backbones_factory import BackbonesFactory


def test_backbones_factory():
    factory = BackbonesFactory()
    assert len(factory.models) > 0

    for name in factory.models.keys():
        model_fn, preprocess_fn, layers = factory.models[name]
        assert callable(model_fn)
        assert callable(preprocess_fn)
        assert isinstance(layers, tuple)


def test_get_backbone_unknown():
    factory = BackbonesFactory()

    with pytest.raises(ValueError):
        factory.get_backbone("unknown")


@pytest.mark.parametrize(
    "backbone_name, expected_shapes",
    [
        (
            "efficientnetv2s-ss",
            ((4, 4, 160), (8, 8, 64), (16, 16, 48), (32, 32, 24)),
        ),
        (
            "efficientnetv2m-ss",
            ((4, 4, 176), (8, 8, 80), (16, 16, 48), (32, 32, 24)),
        ),
        (
            "efficientnetv2l-ss",
            ((4, 4, 224), (8, 8, 96), (16, 16, 64), (32, 32, 32)),
        ),
    ],
)
def test_efficientnetv2_compact_skip_shapes(backbone_name, expected_shapes):
    factory = BackbonesFactory()
    model = factory.get_backbone(
        backbone_name,
        include_top=False,
        weights=None,
        input_shape=(64, 64, 3),
    )

    feature_shapes = tuple(
        tuple(int(size) for size in model.get_layer(layer_name).output.shape[1:])
        for layer_name in factory.get_feature_layers(backbone_name, n=4)
    )

    assert feature_shapes == expected_shapes
