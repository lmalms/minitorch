from typing import List, Tuple, Union

from minitorch.autodiff import Scalar

# Data Types
FloatOrScalar = Union[float, Scalar]


# Metric Types
RocCurveResult = Tuple[List[Scalar], List[Scalar], List[Scalar]]
