from typing import List, Union

from minitorch.autodiff import Scalar, Tensor, TensorBackend, FastOps
from minitorch.module.layer import LinearScalarLayer, LinearTensorLayer
from minitorch.module.module import Module


class ScalarNetwork(Module):
    """
    Builds a network stack of linear fully connected scalar layers,
    each one followed by a relu activation function.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self._input_layer = LinearScalarLayer(input_dim, hidden_dim)
        self._hidden_layer = LinearScalarLayer(hidden_dim, hidden_dim)
        self._output_layer = LinearScalarLayer(hidden_dim, output_dim)

    def forward(self, inputs: List[List[Union[float, Scalar]]]) -> List[List[Scalar]]:
        # Pass through input layer
        input_to_hidden = self._apply_relu(self._input_layer(inputs))

        # Pass through hidden layer
        hidden_to_hidden = self._apply_relu(self._hidden_layer(input_to_hidden))

        # Pass through output layer
        return self._output_layer(hidden_to_hidden)

    @staticmethod
    def _apply_relu(inputs: List[List[Scalar]]) -> List[List[Scalar]]:
        return [[i.relu() for i in sample] for sample in inputs]

    @staticmethod
    def _apply_sigmoid(inputs: List[List[Scalar]]) -> List[List[Scalar]]:
        return [[i.sigmoid() for i in sample] for sample in inputs]


class TensorNetwork(Module):
    """
    Builds network stack of linear fully connected tensor layers,
    each one followed by a relu activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        backend: TensorBackend = TensorBackend(FastOps),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.backend = backend
        self._input_layer = LinearTensorLayer(input_dim, hidden_dim, backend=backend)
        self._hidden_layer = LinearTensorLayer(hidden_dim, hidden_dim, backend=backend)
        self._output_layer = LinearTensorLayer(hidden_dim, output_dim, backend=backend)

    def forward(self, inputs: Tensor) -> Tensor:
        # Pass through input layer
        input_to_hidden = self._apply_relu(self._input_layer(inputs))

        # Pass through hidden layer
        hidden_to_hidden = self._apply_relu(self._hidden_layer(input_to_hidden))

        # Pass through output layer
        return self._output_layer(hidden_to_hidden)

    @staticmethod
    def _apply_relu(inputs: Tensor) -> Tensor:
        return inputs.relu()

    @staticmethod
    def _apply_sigmoid(inputs: Tensor) -> Tensor:
        return inputs.sigmoid()
