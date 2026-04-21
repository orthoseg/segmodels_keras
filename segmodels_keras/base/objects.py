from typing import Any


class KerasObject:
    _backend: Any = None
    _models: Any = None
    _layers: Any = None
    _utils: Any = None

    def __init__(self, name: str | None = None) -> None:
        if (
            self.backend is None
            or self.utils is None
            or self.models is None
            or self.layers is None
        ):
            raise RuntimeError("You cannot use `KerasObjects` with None submodules.")

        self._name = name

    @property
    def __name__(self) -> str:
        if self._name is None:
            return self.__class__.__name__  # type: ignore[return-value]
        return self._name

    @property
    def name(self) -> str:
        return self.__name__

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @classmethod
    def set_submodules(cls, backend: Any, layers: Any, models: Any, utils: Any) -> None:
        cls._backend = backend
        cls._layers = layers
        cls._models = models
        cls._utils = utils

    @property
    def submodules(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "layers": self.layers,
            "models": self.models,
            "utils": self.utils,
        }

    @property
    def backend(self) -> Any:
        return self._backend

    @property
    def layers(self) -> Any:
        return self._layers

    @property
    def models(self) -> Any:
        return self._models

    @property
    def utils(self) -> Any:
        return self._utils


class Metric(KerasObject):
    pass


class Loss(KerasObject):
    def __add__(self, other: Any) -> Any:
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other: Any) -> Any:
        return self.__add__(other)

    def __mul__(self, value: int | float) -> Any:
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError("Loss should be inherited from `BaseLoss` class")

    def __rmul__(self, other: Any) -> Any:
        return self.__mul__(other)


class MultipliedLoss(Loss):
    def __init__(self, loss: Any, multiplier: int | float) -> None:

        # resolve name
        if len(loss.__name__.split("+")) > 1:
            name = f"{multiplier}({loss.__name__})"
        else:
            name = f"{multiplier}{loss.__name__}"
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, gt: Any, pr: Any) -> Any:
        return self.multiplier * self.loss(gt, pr)


class SumOfLosses(Loss):
    def __init__(self, l1: Any, l2: Any) -> None:
        name = f"{l1.__name__}_plus_{l2.__name__}"
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, gt: Any, pr: Any) -> Any:
        return self.l1(gt, pr) + self.l2(gt, pr)
