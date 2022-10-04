from typing import List, Tuple, Union

from typing_extensions import TypeAlias

# from minitorch.autodiff.scalar import Scalar

# Data Types
ScalarLike: TypeAlias = Union[float, int, "Scalar"]
TensorLike: TypeAlias = Union[float, int, "Tensor"]


# Metric Types
# RocCurveResult = Tuple[List[Scalar], List[Scalar], List[Scalar]]
