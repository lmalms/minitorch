from typing import List, Union
import random

from minitorch.module import Module
from minitorch.autodiff import Scalar


class Network(Module):
    """
    Builds a network stack of linear fully connected layers,
    each one followed by a relu activation function.
    """

    def __init__(self, n_hidden_layers: int):
        super().__init__()
        self.n_hidden_layers = n_hidden_layers
        # TODO: intialise linear layer here.

    def forward(self, x: List[Scalar]):
        # TODO: input could also be a list of floats


class Linear(Module):

    """
    Builds a linear fully connected layer.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._weights = []
        self._bias = []

        # Construct the trainable weight matrix using minitorch.Scalars
        # Need a weight matrix of shape (input_dim, output_dim)
        for i in range(input_dim):
            self.weights.append([])
            for j in range(output_dim):
                weight = self.add_parameter(value=Scalar(2 * (random.random() - 0.5)), name=f"weight_{i}_{j}")
                self.weights[i].append(weight)

        # Construct bias matrix using minitorch.Scalars
        # Need a bias term for every output dim
        for j in range(output_dim):
            bias = self.add_parameter(value=Scalar(2 * (random.random() - 0.5)), name=f"bias_{j}")
            self.bias.append(bias)

    def forward(self, inputs: List[List[Union[float, Scalar]]]):
        """
        Forward function for linear layer.
        """

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

