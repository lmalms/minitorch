from typing import List, Union

from minitorch.module.module import Module
from minitorch.module.layer import Linear
from minitorch.autodiff import Scalar


class Network(Module):
    """
    Builds a network stack of linear fully connected layers,
    each one followed by a relu activation function.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            n_hidden_layers: int
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self._input_layer = Linear(input_dim, hidden_dim)
        self._hidden_layers = [Linear(input_dim, hidden_dim) for _ in range(n_hidden_layers)]
        self._output_layer = Linear(hidden_dim, output_dim)

    def forward(self, inputs: List[List[Union[float, Scalar]]]) -> List[List[Scalar]]:
        # Pass through input layer
        hidden_state = self._input_layer.forward(inputs)
        hidden_state = self._apply_relu(hidden_state)

        # Pass through hidden layers
        for layer in self._hidden_layers:
            hidden_state = layer.forward(hidden_state)
            hidden_state = self._apply_relu(hidden_state)

        # Pass through output layer
        out_ = self._output_layer(hidden_state)
        return self._apply_relu(out_)

    @staticmethod
    def _apply_relu(inputs: List[List[Union[Scalar]]]) -> List[List[Scalar]]:
        return [[i.relu() for i in sample] for sample in inputs]
