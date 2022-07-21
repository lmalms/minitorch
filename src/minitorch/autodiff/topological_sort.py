from typing import List

from minitorch.autodiff.variable import Variable


def topological_sort(variable: Variable) -> List[Variable]:
    sorted_ = []

    def dfs_visit(variable: Variable) -> None:

        if variable.is_leaf() or variable.is_constant():
            return

        for v in variable.history.inputs:
            sorted_.append(v)
            dfs_visit(v)

    dfs_visit(variable)

    return sorted_
