from typing import List, Union
import random

from minitorch.module import Module
from minitorch.parameter import Parameter
from minitorch.autodiff import Scalar


class Linear(Module):

    """
    Builds a linear fully connected layer.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._weights = self._initialise_weights(input_dim, output_dim)
        self._bias = self._initialise_bias(output_dim)

    def _initialise_weights(self, input_dim: int, output_dim: int) -> List[List[Parameter]]:
        # Construct the trainable weight matrix using minitorch.Scalars
        weights = []
        for i in range(input_dim):
            # Need a weight matrix of shape (input_dim, output_dim)
            weights.append([])
            for j in range(output_dim):
                weight = self.add_parameter(
                    value=Scalar(2 * (random.random() - 0.5)),
                    name=f"weight_{i}_{j}"
                )
                weights[i].append(weight)

        return weights

    def _initialise_bias(self, output_dim: int) -> List[Parameter]:
        # Construct bias matrix using minitorch.Scalars
        biases = []
        for j in range(output_dim):
            # Need a bias term for every output dim
            bias = self.add_parameter(
                value=Scalar(2 * (random.random() - 0.5)),
                name=f"bias_{j}"
            )
            biases.append(bias)

        return biases

    def forward(self, inputs: List[List[Union[float, Scalar]]]) -> List[List[Scalar]]:
        """
        Forward function for linear layer.
        """

        # TODO: use type hint here for inputs e.g. ScalarBatch

        outputs = []
        for s, sample in enumerate(inputs):
            # First dimension is assumed to be batch size
            outputs.append([])
            for j in range(self.output_dim):
                # Output for index j
                out_ = 0

                # Multiply input features by weights
                weights = [self._weights[i][j] for i in range(self.input_dim)]
                for (feature, weight) in zip(sample, weights):
                    out_ += feature * weight

                # Add bias term
                out_ += self._bias[j]

                # Append to outputs
                outputs[s].append(out_)

        return outputs


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

