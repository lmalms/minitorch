from typing import Union

from minitorch.autodiff import Variable, topological_sort


def backpropagate(variable: Variable, d_out: Union[Variable, float] = 1.0) -> None:
    derivative_chain = topological_sort(variable)
    var_derivative_map = {variable: d_out}

    for i, var in enumerate(derivative_chain):
        if not var.is_leaf():
            # Fetch any derivatives from previous backprop steps
            d_out = var_derivative_map.get(var, 1.0)
            input_diff_pairs = var.history.backprop_step(d_out=d_out)

            # Update scalars with new derivatives
            for (input_, diff) in input_diff_pairs:
                prev_diff = var_derivative_map.get(input_, 0.0)
                var_derivative_map.update({input_: (prev_diff + diff)})
                print(var_derivative_map)

    # Assign derivatives / accumulate derivatices
    for var, derivative in var_derivative_map.items():
        if var.is_leaf():
            var.accumulate_derivative(derivative)
        else:
            var.derivative = derivative
