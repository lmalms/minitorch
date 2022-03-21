from typing import Dict
from parameter import Parameter


class Module:

    """
    Modules form a tree that stores parameters and other submodules. They make up the basis
    of neural network stacks.

    Attributes:
        _modules: dict of name: Module - storage of child modules
        _parameters: dict of name: Parameter - storage of the modules and child modules parameters
        training: bool - whether the module is in training mode.
    """

    def __init__(self):
        self._modules: Dict[str, Module] = {}
        self._parameters: Dict[str, Parameter] = {}
        self.training = True

    def modules(self):
        """
        Returns the module's direct child modules.
        """
        return self.__dict__["_modules"].values()

    def train(self):
        """
        Conditions the module and all descendents to train.
        """
        def train_child_modules(module: Module) -> None:
            module.training = True
            for _, child_module in module._modules.items():
                train_child_modules(module=child_module)

        self.training = True
        for _, child_module in self._modules.items():
            train_child_modules(module=child_module)
