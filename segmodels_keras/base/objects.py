"""Base objects for segmodels_keras."""

from typing import Any


class KerasObject:
    """Base class for all objects in segmodels_keras."""

    _backend: Any = None
    _models: Any = None
    _layers: Any = None
    _utils: Any = None

    def __init__(self, name: str | None = None) -> None:  # noqa: D107
        if (
            self.backend is None
            or self.utils is None
            or self.models is None
            or self.layers is None
        ):
            raise RuntimeError("You cannot use `KerasObjects` with None submodules.")

        self._name = name

    @property
    def __name__(self) -> str:  # noqa: D105
        if self._name is None:
            return self.__class__.__name__  # type: ignore[return-value]
        return self._name

    @property
    def name(self) -> str:  # noqa: D102
        return self.__name__

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @classmethod
    def set_submodules(cls, backend: Any, layers: Any, models: Any, utils: Any) -> None:  # noqa: D102
        cls._backend = backend
        cls._layers = layers
        cls._models = models
        cls._utils = utils

    @property
    def submodules(self) -> dict[str, Any]:  # noqa: D102
        return {
            "backend": self.backend,
            "layers": self.layers,
            "models": self.models,
            "utils": self.utils,
        }

    @property
    def backend(self) -> Any:  # noqa: D102
        return self._backend

    @property
    def layers(self) -> Any:  # noqa: D102
        return self._layers

    @property
    def models(self) -> Any:  # noqa: D102
        return self._models

    @property
    def utils(self) -> Any:  # noqa: D102
        return self._utils


class Metric(KerasObject):
    """Base class for all metrics."""


class Loss(KerasObject):
    """Base class for all loss functions."""

    def __add__(self, other: Any) -> Any:  # noqa: D105
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other: Any) -> Any:  # noqa: D105
        return self.__add__(other)

    def __mul__(self, value: int | float) -> Any:  # noqa: D105
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError("Loss should be inherited from `BaseLoss` class")

    def __rmul__(self, other: Any) -> Any:  # noqa: D105
        return self.__mul__(other)


class MultipliedLoss(Loss):
    """Class for multiplying loss by a constant value."""

    def __init__(self, loss: Any, multiplier: int | float) -> None:  # noqa: D107

        # resolve name
        if len(loss.__name__.split("+")) > 1:
            name = f"{multiplier}({loss.__name__})"
        else:
            name = f"{multiplier}{loss.__name__}"
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, gt: Any, pr: Any) -> Any:  # noqa: D102
        return self.multiplier * self.loss(gt, pr)


class SumOfLosses(Loss):
    """Class for summing two losses.

    For example, you can sum DiceLoss and BinaryCELoss.
    """

    def __init__(self, l1: Any, l2: Any) -> None:  # noqa: D107
        name = f"{l1.__name__}_plus_{l2.__name__}"
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, gt: Any, pr: Any) -> Any:  # noqa: D102
        return self.l1(gt, pr) + self.l2(gt, pr)
