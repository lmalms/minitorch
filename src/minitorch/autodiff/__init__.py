from minitorch.autodiff.scalar import (
    Scalar,
    ScalarFunction,
    central_difference,
    derivative_check,
)
from minitorch.autodiff.tensor import Tensor
from minitorch.autodiff.tensor_data import TensorData, shape_broadcast
from minitorch.autodiff.tensor_functions import (
    FastOps,
    SimpleOps,
    TensorBackend,
    TensorFunction,
    grad_check,
    ones,
    rand,
    tensor,
    zeros,
)
from minitorch.autodiff.variable import (
    Context,
    History,
    Variable,
    backpropagate,
    topological_sort,
)

FastBackend = TensorBackend(FastOps)
SimpleBackend = TensorBackend(SimpleOps)
