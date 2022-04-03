from __future__ import annotations
from typing import Dict, List, Tuple, Union
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

    def __setattr__(self, key: str, value: Union[Parameter, Module]):
        if isinstance(value, Parameter):
            self.__dict__["_parameters"][key] = value
        elif isinstance(value, Module):
            self.__dict__["_modules"][key] = value
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key: str):
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]
        elif key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):

        def _add_indent(s_: str, numSpaces: int):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = "\n".join([(" " * numSpaces) + line for line in s])
            return first + "\n" + s

        child_lines = []
        for key, module in self.__dict__["_modules"].items():
            module_str = repr(module)
            module_str = _add_indent(module_str, numSpaces=2)
            child_lines.append("(" + key + "): " + module_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            main_str += "\n " + "\n ".join(lines) + "\n"
        main_str += ")"
        return main_str

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
        def add_child_parameters(
                parent_parameters: List[Tuple[str, Parameter]],
                child_module: Module,
                prefix: str
        ) -> List[Tuple[str, Parameter]]:
            new_params = [
                (prefix + name, param) for (name, param) in child_module.__dict__["_parameters"].items()
            ]
            parent_parameters.extend(new_params)
            for child_module in child_module.modules():
                parent_parameters = add_child_parameters(
                    parent_parameters=parent_parameters,
                    child_module=child_module,
                    prefix=prefix + child_module.__name__
                )
            return parent_parameters

        named_params = [(self.__name__ + name, param) for name, param in self.__dict__["_parameters"].items()]
        for child_module in self.modules():
            named_params = add_child_parameters(
                parent_parameters=named_params,
                child_module=child_module,
                prefix=self.__name__ + child_module.__name__
            )
        return named_params

    def parameters(self) -> List[Parameter]:
        """
        Enumerates over all parameters of this module and its descendants.
        """
        params = [param for (_, param) in self.__dict__["_parameters"].items()]
        for child_module in self.modules():
            params.extend(child_module.parameters())
        return params

    def add_parameter(self, parameter: Parameter) -> Parameter:
        self.__dict__["_parameters"][parameter.name] = parameter.value
        return parameter

    def forward(self, *args, **kwargs):
        assert False, "Not implemented."




