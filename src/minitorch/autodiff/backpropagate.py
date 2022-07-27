from minitorch.autodiff import Variable, topological_sort
from typing import Union


def backpropagate(variable: Variable, d_out: Union[float, Variable]) -> None:
    derivative_chain = topological_sort(variable)
    vars_to_derivatives = {}

    ###
    '''
    1. Call topological sort to get an ordered queue

    2. Create a dictionary of Variables and current derivatives

    3. For each node in backward order, pull a completed Variable and derivative from the queue:

if the Variable is a leaf, add its final derivative (accumulate_derivative) and loop to (1)

if the Variable is not a leaf,

call .backprop_step on the last function that created it with derivative as 𝑑𝑜𝑢𝑡

loop through all the Variables+derivative produced by the chain rule

accumulate derivatives for the Variable in a dictionary (check .unique_id)
'''
    ###