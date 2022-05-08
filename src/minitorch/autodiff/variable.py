from __future__ import annotations
from typing import Optional, List

from minitorch.autodiff.utils import unwrap_tuple

VARIABLE_COUNT = 1


class Context:
    """
    The context class is used by Functions to store information during the forward pass.

    Attributes:
        no_grad - bool
            Whether to save gradient information or not.
        saved_values - Tuple[]
            A tuple of values saved for backward pass.
        saved_tensors - Tuple[]
            A tuple of tensors saved for backward pass - alias for saved_values.

    """

    def __init__(self, no_grad: bool = False):
        self._saved_values = None
        self.no_grad = no_grad

    @property
    def saved_values(self):
        assert not self.no_grad, "Doesn't require grad."
        assert self._saved_values is not None, "Did you forget to save values?"
        return unwrap_tuple(self._saved_values)

    @property
    def saved_tensors(self):
        return self.saved_values

    def save_for_backward(self, *values):
        """
        Stores the given values if they need to be used during back-propagation
        """
        if self.no_grad:
            return None
        self._saved_values = values


class History:
    """
    History stores the history of Function operations that were used to create
    a Variable.

    Attributes:
        last_fn - Function
            The last function that was called
        ctx - Context
            The context for that function.
        inputs - List
            The inputs that were given when last_fn.forward was called.
    """
    def __init__(
            self,
            last_fn: Optional[BaseFunction] = None,
            ctx: Optional[Context] = None,
            inputs: Optional[List[float]] = None
    ):
        self.last_fn = last_fn
        self.ctx = ctx
        self.inputs = inputs

    def backprop_step(self, d_output) -> List[float]:
        """
        Runs one step of back-propagation by calling the chain rule.

        Args:
            d_output -
                a derivative w.r.t. a given Variable

        Returns:
            List[Any]
                The derivatives w.r.t. inputs.
        """
        # TODO: implement this.
        raise NotImplementedError


class BaseFunction:
    pass



class Variable:
    """
    Class for tracking variable values and computation history for auto-differentiation.

    Attributes:
        history
        derivative
        grad
        name

    Attributes cannot be manipulated directly, only through the use of functions that act on the variable.
    """
    def __init__(self, history: History, name: Optional[str] = None):
        global VARIABLE_COUNT
        assert history is None or isinstance(history, History)

        self.history = history
        self._derivative = None

        VARIABLE_COUNT += 1
        self.id = "Variable" + str(VARIABLE_COUNT)
        self.name = name if name is not None else self.id
        self.used = 0

    def requires_grad_(self, val: bool):
        """
        Sets the requires_grad_ flag to 'val' on variable. This ensures that operations on this variable will trigger
        backpropagation.
        """
        self.history = History()

