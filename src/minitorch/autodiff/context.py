from minitorch.autodiff.utils import unwrap_tuple


class Context:
    """
    The context class is used by Functions to store information during the forward pass, which in turn is needed
    for computations during the backward pass.

    Attributes:
        requires_grad_ - bool
            Whether to save gradient information or not.
        saved_values - Tuple[]
            A tuple of values saved for backward pass.
        saved_tensors - Tuple[]
            A tuple of tensors saved for backward pass - alias for saved_values.

    """

    def __init__(self, requires_grad_: bool = False):
        self.requires_grad_ = requires_grad_
        self._saved_values = None

    @property
    def saved_values(self):
        assert self.requires_grad_, "No gradients required - no values saved."
        assert (
            self._saved_values is not None
        ), "No values saved - did you forget to save values?"
        return unwrap_tuple(self._saved_values)

    @property
    def saved_tensors(self):
        return self.saved_values

    def save_for_backward(self, *values) -> None:
        """
        Stores the given values if they need to be used during back-propagation.
        """
        if self.requires_grad_:
            self._saved_values = values
