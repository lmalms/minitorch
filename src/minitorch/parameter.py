from typing import Optional


class Parameter:

    def __init__(self, value: Optional[float] = None, name: Optional[str] = None):
        self.value = value
        self.name = name
        self.update(value=value)

    def update(self, value: Optional[float] = None):
        """
        Updates the parameters value.
        """
        self.value = value
        if hasattr(value, "requires_grad_"):
            self.value.requires_grad_(True)  # TODO: need to implement a value class
            if self.name:
                self.value.name = self.name

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)
