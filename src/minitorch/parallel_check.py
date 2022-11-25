from numba import njit

import minitorch.autodiff.fast_tensor_ops as fast_ops
import minitorch.autodiff.tensor_functions as tf
import minitorch.operators as ops

# MAP
print("MAP")
tmap = fast_ops.tensor_map(njit()(ops.identity))
out_, in_ = tf.zeros((10,)), tf.zeros((10,))
tmap(*out_.tuple(), *in_.tuple())
print(tmap.parallel_diagnostics(level=3))
