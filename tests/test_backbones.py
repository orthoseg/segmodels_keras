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
