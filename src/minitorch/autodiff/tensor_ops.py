from __future__ import annotations

from typing import Any, Callable, Optional, Type

import numpy as np

from minitorch import operators

from .tensor_data import (
    Storage,
    _Index,
    _Shape,
    _Strides,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
