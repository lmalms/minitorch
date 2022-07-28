from typing import List, Union

from minitorch.autodiff.variable import Variable


def topological_sort(variable: Union[Variable, float]) -> List[Variable]:
    diff_chain = []

    def dfs_visit(variable: Variable) -> None:

        if not isinstance(variable, Variable):
            return

        if variable.is_constant():
            return

        if hasattr(variable, "seen") and variable.seen:
            return

        # Append variable to list
        diff_chain.append(variable)

        # Mark variable as visited
        setattr(variable, "seen", True)

        # Iterate over variables descendants
        if variable.history.inputs is not None:
            for v in variable.history.inputs:
                dfs_visit(v)

    def visit_variable_and_add(variable):
        if not isinstance(variable, Variable):
            return

        if variable.is_constant():
            return

        if hasattr(variable, "seen") and variable.seen:
            return

        # Append variable to list
        diff_chain.append(variable)

        # Mark variable as visited
        setattr(variable, "seen", True)

    def bfs_visit(variable: Variable) -> None:

        # Append all of its children to list
        if variable.history.inputs is not None:
            for v in variable.history.inputs:
                visit_variable_and_add(v)

            # Run bfs_visit on children
            for v in variable.history.inputs:
                bfs_visit(v)

    def remove_seen(variables: List[Variable]) -> None:
        for variable in variables:
            if hasattr(variable, "seen"):
                delattr(variable, "seen")

    visit_variable_and_add(variable)
    bfs_visit(variable)
    remove_seen(diff_chain)
    return diff_chain
