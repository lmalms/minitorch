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


# ZIP
print("ZIP")
out, a, b = tf.zeros((10,)), tf.zeros((10,)), tf.zeros((10,))
tzip = fast_ops.tensor_zip(njit()(ops.eq))
tzip(*out.tuple(), *a.tuple(), *b.tuple())
print(tzip.parallel_diagnostics(level=3))


# REDUCE
print("REDUCE")
out, a = tf.zeros((1,)), tf.zeros((10,))
treduce = fast_ops.tensor_reduce(njit()(ops.add))
treduce(*out.tuple(), *a.tuple(), 0)
print(treduce.parallel_diagnostics(level=3))

# MATMUL
print("MATMUL")
out, a, b = (
    tf.zeros((1, 10, 10)),
    tf.zeros((1, 10, 20)),
    tf.zeros((1, 20, 10)),
)
tmatmul = fast_ops.tensor_matrix_multiply

tmatmul(*out.tuple(), *a.tuple(), *b.tuple())
print(tmatmul.parallel_diagnostics(level=3))
