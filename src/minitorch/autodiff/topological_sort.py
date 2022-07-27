from typing import List, Union

from minitorch.autodiff.variable import Variable


def topological_sort(variable: Union[Variable, float]) -> List[Variable]:
    sorted_ = []

    def dfs_visit(variable: Variable) -> None:

        if variable.is_constant() or (variable.history.inputs is None):
            return

        for v in variable.history.inputs:
            if not isinstance(v, Variable):
                continue

            if hasattr(v, "seen") and v.seen:
                continue

            # Append variable to list
            sorted_.append(v)

            # Mark variable as visited
            setattr(v, "seen", True)

            # Iterate over variables descendants
            dfs_visit(v)

    def remove_seen(variables: List[Variable]) -> None:
        for variable in variables:
            if hasattr(variable, "seen"):
                delattr(variable, "seen")

    if not isinstance(variable, Variable):
        return sorted_

    dfs_visit(variable)
    remove_seen(sorted_)
    return sorted_
