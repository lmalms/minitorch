from typing import List, Tuple, Union

from minitorch.autodiff import Scalar

# Data Types
ScalarLike = Union[float, int, "Scalar"]
TensorLike = Union[float, int, "Tensor"]


# Metric Types
RocCurveResult = Tuple[List[Scalar], List[Scalar], List[Scalar]]
