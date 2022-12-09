import itertools
import random
import time
from typing import List, Optional, Union

import minitorch.autodiff.tensor_functions as tf
import minitorch.scalar_losses as sl
import minitorch.tensor_losses as tl
from minitorch.autodiff import FastOps, Scalar, Tensor, TensorBackend
from minitorch.module.module import Module
from minitorch.module.parameter import Parameter
from minitorch.optim.base import BaseOptimizer

from .logging import default_message, logging

logger = logging.getLogger(__name__)


class LinearScalarLayer(Module):

    """
    Builds a linear fully connected layer using scalar variables.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._weights = self._initialise_weights(input_dim, output_dim)
        self._bias = self._initialise_bias(output_dim)

    def _initialise_weights(
        self, input_dim: int, output_dim: int
    ) -> List[List[Parameter]]:
        # Construct the trainable weight matrix using minitorch.Scalars
        weights = []
        for i in range(input_dim):
            # Need a weight matrix of shape (input_dim, output_dim)
            weights.append([])
            for j in range(output_dim):
                weight = self.add_parameter(
                    value=Scalar(2 * (random.random() - 0.5)), name=f"weight_{i}_{j}"
                )
                weights[i].append(weight)

        return weights

    def _initialise_bias(self, output_dim: int) -> List[Parameter]:
        # Construct bias matrix using minitorch.Scalars
        biases = []
        for j in range(output_dim):
            # Need a bias term for every output dim
            bias = self.add_parameter(
                value=Scalar(2 * (random.random() - 0.5)), name=f"bias_{j}"
            )
            biases.append(bias)

        return biases

    def forward(self, inputs: List[List[Union[float, Scalar]]]) -> List[List[Scalar]]:
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
                    out_ += feature * weight.value

                # Add bias term
                out_ += self._bias[j].value

                # Append to outputs
                outputs[s].append(out_)

        return outputs

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
        for epoch in range(1, n_epochs + 1):
            start_time = time.time()

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

            # Update parameters
            optimizer.step()

            # Record
            time_per_epoch = (time.time() - start_time) * 1e03
            losses.append(loss_per_epoch.data)
            if epoch % logging_freq == 0:
                logger.info(
                    default_message(
                        epoch,
                        loss_per_epoch.data,
                        time_per_epoch,
                    )
                )

        return losses


class LinearTensorLayer(Module):

    """
    Builds a linear fully connected layer using tensor variables.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        backend: TensorBackend = TensorBackend(FastOps),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.backend = backend
        self._weights = self._initialise_parameter(
            input_dim,
            output_dim,
            backend=backend,
            name="linear_weight",
        )
        self._bias = self._initialise_parameter(
            output_dim,
            backend=backend,
            name="linear_bias",
        )

    @staticmethod
    def _initialise_parameter(
        *shape: int,
        backend: TensorBackend,
        name: Optional[str] = None,
    ) -> Parameter:
        random_tensor = tf.rand(
            shape=tuple(shape),
            requires_grad=True,
            backend=backend,
        )
        random_tensor = 2 * (random_tensor - 0.5)
        return Parameter(value=random_tensor, name=name)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Implements forward function for tensors using broadcasting.
        """
        # Check inputs and set same backend
        assert inputs.shape[1] == self._weights.value.shape[0]
        inputs._type_(backend=self.backend)

        # Add dimensions such that we can broadcast
        _inputs = inputs.view(*inputs.shape, 1)
        _weights = self._weights.value.view(1, *self._weights.value.shape)

        # Collapse dimension
        _out = (_inputs * _weights).sum(dim=1)
        _out = _out.view(inputs.shape[0], self.output_dim)
        return _out + self._bias.value

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
        for epoch in range(1, n_epochs + 1):
            start_time = time.time()
            # Zero all grads
            optimizer.zero_grad()

            # Forward
            y_hat = self.forward(features).sigmoid()

            # Compute a loss
            loss_per_epoch = tl.binary_cross_entropy(labels, y_hat)
            loss_per_epoch.backward()

            # Update parameters
            optimizer.step()

            # Record
            # Record
            time_per_epoch = (time.time() - start_time) * 1e03
            losses.append(loss_per_epoch.item())
            if epoch % logging_freq == 0:
                logger.info(
                    default_message(
                        epoch,
                        loss_per_epoch.item(),
                        time_per_epoch,
                    )
                )

        return losses
