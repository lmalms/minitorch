from typing import List, Tuple

from minitorch.autodiff import Scalar, summation
from minitorch.functional import EPS, operators


class MathTestOperators:
    @staticmethod
    def neg(x: float) -> float:
        return -x

    @staticmethod
    def add_constant(x: float) -> float:
        return x + 5.0

    @staticmethod
    def subtract_constant(x: float) -> float:
        return x - 5.0

    @staticmethod
    def multiply(x: float) -> float:
        return 5.0 * x

    @staticmethod
    def divide(x: float) -> float:
        return x / 5.0

    @staticmethod
    def square(x: float):
        return x * x

    @staticmethod
    def cube(x: float):
        return x * x * x

    @staticmethod
    def sigmoid(x: float) -> float:
        return operators.sigmoid(x)

    @staticmethod
    def inv(x: float):
        return operators.inv(x + 3.5)

    @staticmethod
    def log(x: float) -> float:
        return operators.log(abs(x) + EPS)  # To assure values are positive for tests.

    @staticmethod
    def relu(x: float) -> float:
        return operators.relu(x + 5.5)

    @staticmethod
    def exp(x: float) -> float:
        return operators.exp(x - 200)

    @staticmethod
    def explog(x: float) -> float:
        return operators.log(x + 100000) + operators.exp(x - 200)

    @staticmethod
    def add2(x: float, y: float) -> float:
        return x + y

    @staticmethod
    def multiply2(x: float, y: float) -> float:
        return x * y

    @staticmethod
    def divide2(x: float, y: float) -> float:
        return x / (y + 5.5)

    @staticmethod
    def gt2(x: float, y: float) -> float:
        return operators.lt(y, x + 1.2)

    @staticmethod
    def lt2(x: float, y: float) -> float:
        return operators.lt(x + 1.2, y)

    @staticmethod
    def eq2(x: float, y: float) -> float:
        return operators.eq(x, (y + 5.5))

    @staticmethod
    def summation_reduction(x: List[float]) -> float:
        return operators.summation(x)

    @staticmethod
    def mean_reduction(x: List[float]) -> float:
        return operators.summation(x) / float(len(x))

    @staticmethod
    def complex(x: float):
        return (
            operators.log(
                operators.sigmoid(
                    operators.relu(operators.relu(x * 10 + 7) * 6 + 5) * 10
                )
            )
            / 50
        )

    @classmethod
    def generate_tests(cls) -> Tuple[List, List, List]:
        """
        Collates all tests.
        """
        one_arg_tests = []
        two_arg_tests = []
        reduction_tests = []
        for k in dir(MathTestOperators):
            if callable(getattr(MathTestOperators, k)) and not (
                k.startswith("generate") or k.startswith("_")
            ):
                base_fn = getattr(MathTestOperators, k)
                scalar_fn = getattr(cls, k)
                tup = (k, base_fn, scalar_fn)
                if k.endswith("2"):
                    two_arg_tests.append(tup)
                elif k.endswith("reduction"):
                    reduction_tests.append(tup)
                else:
                    one_arg_tests.append(tup)

        return one_arg_tests, two_arg_tests, reduction_tests


class MathTestVariable(MathTestOperators):
    @staticmethod
    def inv(x: Scalar):
        return 1.0 / (x + 3.5)

    @staticmethod
    def square(x: Scalar):
        return x.square()

    @staticmethod
    def cube(x: Scalar):
        return x.cube()

    @staticmethod
    def sigmoid(x: Scalar):
        return x.sigmoid()

    @staticmethod
    def log(x: Scalar):
        return (x + 1e06).log()

    @staticmethod
    def relu(x: Scalar):
        return (x + 5.5).relu()

    @staticmethod
    def exp(x: Scalar):
        return (x - 200.0).exp()

    @staticmethod
    def explog(x: Scalar):
        return (x + 1e06).log() + (x - 200.0).exp()

    @staticmethod
    def add2(x: Scalar, y: Scalar) -> Scalar:
        return x + y

    @staticmethod
    def multiply2(x: Scalar, y: Scalar) -> Scalar:
        return x * y

    @staticmethod
    def divide2(x: Scalar, y: Scalar) -> Scalar:
        return x / (y + 5.5)

    @staticmethod
    def gt2(x: Scalar, y: Scalar) -> Scalar:
        return (x + 1.2) > y

    @staticmethod
    def lt2(x: Scalar, y: Scalar) -> Scalar:
        return (x + 1.2) < y

    @staticmethod
    def eq2(x: Scalar, y: Scalar) -> Scalar:
        return x == (y + 5.5)

    @staticmethod
    def summation_reduction(x: List[Scalar]) -> Scalar:
        return summation(x)

    @staticmethod
    def mean_reduction(x: List[Scalar]) -> Scalar:
        return summation(x) / float(len(x))

    @staticmethod
    def complex(x: Scalar) -> Scalar:
        return (((x * 10 + 7).relu() * 6 + 5).relu() * 10).sigmoid().log() / 50
