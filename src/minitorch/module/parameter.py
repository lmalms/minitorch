from typing import Optional, Union

from minitorch.autodiff import Variable


class Parameter:
    def __init__(
        self, value: Optional[Union[float, Variable]] = None, name: Optional[str] = None
    ):
        self.value = value
        self.name = name
        self.update(value=value)

    @property
    def derivative(self) -> float:
        """
        Returns the parameter's value's derivative
        """
        if hasattr(self.value, "derivative"):
            return self.value.derivative
        return 0.0

    def update(self, value: Optional[Union[float, Variable]] = None) -> None:
        """
        Updates the parameter's value.
        """
        self.value = value
        if hasattr(self.value, "requires_grad"):
            self.value.requires_grad_(True)
            if self.name is not None:
                self.value.name = self.name

    def zero_grad(self) -> None:
        if hasattr(self.name, "zero_grad_"):
            self.value.zero_grad_()

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)
