from __future__ import annotations
from typing import Dict, List, Tuple
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
        self.training: bool = True

    def modules(self) -> List[Module]:
        """
        Returns the module's direct child modules.
        """
        return self.__dict__["_modules"].values()

    def train(self) -> None:
        """
        Conditions the module and all descendants to train (training = True).
        """
        self.training = True
        for child_module in self.modules():
            child_module.train()

    def eval(self) -> None:
        """
        Conditions the module and all descendant modules to eval (training = False).
        """
        self.training = False
        for child_module in self.modules():
            child_module.eval()

    def named_parameters(self) -> List[Tuple[str, Parameter]]:
        """
        Collects all the named parameters of the module and its descendants.
        Returns: List[Tuple[str, Parameter]]
        """
        named_params = [(name, parameter) for name, parameter in self.__dict__["_parameters"].items()]
        for child_modules in self.modules():
            named_params.extend(child_modules.named_parameters())
        return named_params

    def parameters(self) -> List[Parameter]:
        """
        Enumerates over all parameters of this module and its descendants.
        """
        params = [param for (_, param) in self.__dict__["_parameters"].items()]
        for child_module in self.modules():
            params.extend(child_module.parameters())
        return params
