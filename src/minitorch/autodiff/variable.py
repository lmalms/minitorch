from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional, Type, Union

from minitorch.autodiff.utils import unwrap_tuple, wrap_tuple

VARIABLE_COUNT = 1


class Context:
    """
    The context class is used by Functions to store information during the forward pass,
    which in turn is needed for computations during the backward pass.

    Attributes:
        requires_grad_ - bool
            Whether to save gradient information or not.
        saved_values - Tuple[]
            A tuple of values saved for backward pass.
        saved_tensors - Tuple[]
            A tuple of tensors saved for backward pass - alias for saved_values.

    """

    def __init__(self, requires_grad_: bool = False):
        self.requires_grad_ = requires_grad_
        self._saved_values = None

    @property
    def saved_values(self):

        if not self.requires_grad_:
            raise ValueError("Context does not require gradients - no values saved.")

        if self._saved_values is None:
            raise ValueError("No values saved - forgot to run save values?")

        # TODO: Do I need this wrapping and unwrapping here?
        #  when accessing the context we should know how many values we need.
        return unwrap_tuple(self._saved_values)

    @property
    def saved_tensors(self):
        return self.saved_values

    def save_for_backward(self, *values) -> None:
        """
        Stores the given values if they need to be used during back-propagation.
        """
        if self.requires_grad_:
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
        last_fn: Optional[Type[BaseFunction]] = None,
        ctx: Optional[Context] = None,
        inputs: Optional[
            List[float]
        ] = None,  # TODO: this should probs be union of float and variable
    ):
        self.last_fn = last_fn
        self.ctx = ctx
        self.inputs = inputs

    def backprop_step(self, d_out: float) -> List[float]:
        """
        Runs one step of back-propagation by calling the chain rule.
        """
        var_derivative_pairs = self.last_fn.chain_rule(
            self.ctx, self.inputs, d_out=d_out
        )
        derivatives = [derivative for (_, derivative) in var_derivative_pairs]
        return derivatives


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

    def __init__(self, history: Optional[History] = None, name: Optional[str] = None):
        self.history = history
        self.id = self._format_variable_id()
        self.name = name if name is not None else self.id
        self._derivative = None
        self.used = 0

    @property
    @abstractmethod
    def data(self):
        ...

    @property
    def history(self) -> Optional[History]:
        return self._history

    @history.setter
    def history(self, history: Optional[History] = None):
        if not ((history is None) or isinstance(history, History)):
            raise TypeError(
                f"History has to be None or of type history - got {type(history)}"
            )
        self._history = history

    @property
    def derivative(self):
        return self._derivative

    @staticmethod
    def _format_variable_id() -> str:
        global VARIABLE_COUNT
        VARIABLE_COUNT += 1
        return "Variable" + str(VARIABLE_COUNT)

    def is_leaf(self):
        """True if this variable has no last_fn"""
        return self.history.last_fn is None

    def is_constant(self):
        """True if this variable has no history."""
        return self.history is None

    def requires_grad_(self, val: bool):
        """
        Sets the requires_grad_ flag to 'val' on variable.
        This ensures that operations on this variable will trigger
        backpropagation.
        """
        self.history = History()

    def backward(self, d_out: float = 1.0):
        """
        Calls auto-diff to fill in the derivatives for the history of this object.

        Args:
            d_output - float, default = 1.
                Starting derivative to backpropagate through the model
        """
        backpropagate(self, d_out)

    def accumulate_derivative(self, val: float):
        """
        Add val to the derivative accumulated on this variable.
        Should only be called during auto-differentiation on leaf variables.

        Args:
            val - float
                The value to be accumulated.
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self._derivative = self.zeros()
        self._derivative += val

    def zero_derivative_(self) -> None:
        """
        Reset derivative on this variable.
        """
        self._derivative = self.zeros()

    def zero_grad_(self) -> None:
        self.zero_derivative_()

    def expand(self, x):
        """
        Placeholder for tensor variables.
        """
        return x

    @staticmethod
    def zeros() -> float:
        return 0.0


class BaseFunction:
    """
    Base class for functions that act on Variables to produce a new Variable output
    whilst keeping track of the variable's history.

    Called using apply() method.
    """

    @classmethod
    @abstractmethod
    def data_type(cls):
        ...

    @classmethod
    @abstractmethod
    def variable(cls, value, history: History):
        ...

    @staticmethod
    def is_constant(value: Union[Variable, float]) -> bool:
        # TODO: is this the best place to put this?
        # TODO: also used in topological sort
        return not isinstance(value, Variable) or value.history is None

    @classmethod
    @abstractmethod
    def forward(cls, ctx: Context, *values):
        """
        To be implemented by all inheriting Function classes.
        Returns a value of type cls.data_type
        """
        ...

    @classmethod
    @abstractmethod
    def backward(cls, ctx: Context, d_out: float):
        ...

    @classmethod
    def chain_rule(
        cls, ctx: Context, inputs: List[Union[Variable, float]], d_out: float
    ):
        """
        Implements the chain rule for differentiation.
        """
        derivatives = wrap_tuple(cls.backward(ctx, d_out))
        var_dev_pairs = list(zip(inputs, derivatives))
        var_dev_pairs = [
            pair for pair in var_dev_pairs if not cls.is_constant(value=pair[0])
        ]
        return var_dev_pairs

    @classmethod
    def apply(cls, *variables: Union[Variable, float]) -> Variable:
        """
        Apply is used to run the function.
        Internally it does three things:
        a) Create context for the function call
        b) Calls forward to run the function.
        c) Attaches the context to the history of the new variable.

        Args:
            variables - List[Union[Variable, float]]
                An iterable of variables or constants to call forward on.

        Returns:
            Variable
                The new computed variable.
        """
        # Extract raw values
        raw_values = []
        need_grad = False
        for v in variables:
            if isinstance(v, Variable):
                if v.history is not None:
                    need_grad = True
                v.used += 1
                raw_values.append(v.data)
            else:
                raw_values.append(v)

        # Create context
        ctx = Context(need_grad)

        # Call forward with variables
        c = cls.forward(ctx, *raw_values)
        if not isinstance(c, cls.data_type()):
            raise TypeError(f"Expected return type {cls.data_type()}, got {type(c)}.")

        # Create new variable from result with new history.
        back = None
        if need_grad:
            back = History(last_fn=cls, ctx=ctx, inputs=variables)

        return cls.variable(cls.data_type(c), back)
