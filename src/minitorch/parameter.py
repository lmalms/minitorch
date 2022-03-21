from typing import Optional


class Parameter:

    def __init__(self, value=None, name: Optional[str] = None):  # TODO: add type info for value
        self.value = value
        self.name = name
        self.update(value=value)

    def update(self, value):
        """
        Updates the parameters value.
        """
        self.value = value
        if hasattr(value, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)
