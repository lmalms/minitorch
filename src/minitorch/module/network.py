from typing import List, Union

from minitorch.autodiff import Scalar
from minitorch.module.layer import Linear
from minitorch.module.module import Module


class Network(Module):
    """
    Builds a network stack of linear fully connected layers,
    each one followed by a relu activation function.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, n_hidden_layers: int
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self._input_layer = Linear(input_dim, hidden_dim)
        self._hidden_layer = Linear(hidden_dim, hidden_dim)
        self._output_layer = Linear(hidden_dim, output_dim)

    def forward(self, inputs: List[List[Union[float, Scalar]]]) -> List[List[Scalar]]:
        # Pass through input layer
        input_to_hidden = self._apply_relu(self._input_layer.forward(inputs))

        # Pass through hidden layer
        hidden_to_hidden = self._apply_relu(self._hidden_layer.forward(input_to_hidden))

        # Pass through output layer
        hidden_to_output = self._apply_relu(self._output_layer(hidden_to_hidden))
        return hidden_to_output

    @staticmethod
    def _apply_relu(inputs: List[List[Union[Scalar]]]) -> List[List[Scalar]]:
        return [[i.relu() for i in sample] for sample in inputs]
