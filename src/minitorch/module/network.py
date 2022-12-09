import itertools
from typing import List, Union

import minitorch.scalar_losses as sl
import minitorch.tensor_losses as tl
from minitorch.autodiff import FastOps, Scalar, Tensor, TensorBackend
from minitorch.module.layer import LinearScalarLayer, LinearTensorLayer
from minitorch.module.module import Module
from minitorch.optim.base import BaseOptimizer


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

    def fit_binary_classifier(
        self,
        features: List[List[Union[float, Scalar]]],
        labels: List[Union[float, Scalar]],
        optimizer: BaseOptimizer,
        n_epochs: int = 200,
        logging_freq: int = 10,
    ) -> List[float]:
        """
        Trains the parameters of the module using a binary cross entropy objective
        """

        # Check dims
        assert len(features) == len(labels)
        assert all(len(feature) == self.input_dim for feature in features)

        # Training loop
        losses = []
        for epoch in range(n_epochs):
            # Zero all grads
            optimizer.zero_grad()

            # Forward
            y_hat = self.forward(features)

            # Convert to binary class probabilties
            y_hat = [[scalar.sigmoid() for scalar in row] for row in y_hat]
            y_hat = list(itertools.chain.from_iterable(y_hat))

            # Compute a loss
            loss_per_epoch = sl.binary_cross_entropy(labels, y_hat)
            loss_per_epoch.backward()

            optimizer.step()

            # Record
            losses.append(loss_per_epoch.data)
            if epoch % logging_freq == 0:
                print(f"epoch {epoch}: loss = {loss_per_epoch.data}")

            if epoch + 1 == n_epochs:
                print(f"epoch {epoch + 1}: loss = {loss_per_epoch.data}")

        return losses


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

    def fit_binary_classifier(
        self,
        features: Tensor,
        labels: Tensor,
        optimizer: BaseOptimizer,
        n_epochs: int = 200,
        logging_freq: int = 10,
    ) -> List[float]:

        # Check dims
        assert features.dims == 2
        assert labels.dims == 2
        assert features.shape[1] == self.input_dim
        assert features.shape[0] == labels.shape[0]

        # Training loop
        losses = []
        for epoch in range(n_epochs):

            # Zero all grads
            optimizer.zero_grad()

            # Forward
            y_hat = self.forward(features).sigmoid()

            # Compute a loss
            loss_per_epoch = tl.binary_cross_entropy(labels, y_hat)
            loss_per_epoch.backward()

            optimizer.step()

            # Record
            losses.append(loss_per_epoch.item())
            if epoch % logging_freq == 0:
                print(f"epoch {epoch}: loss = {loss_per_epoch.item()}")

            if epoch == (n_epochs - 1):
                print(f"epoch {epoch + 1}: loss = {loss_per_epoch.item()}")

        return losses
