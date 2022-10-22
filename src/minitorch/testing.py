from __future__ import annotations

from typing import Callable, Generic, Iterable, List, Tuple, TypeVar

from minitorch import operators
from minitorch.autodiff import Scalar, Tensor
from minitorch.constants import EPS
from minitorch.functional import summation

T = TypeVar("T")


class MathTest(Generic[T]):
    @staticmethod
    def neg(x: T) -> T:
        return -x

    @staticmethod
    def add_constant(x: T) -> T:
        return x + 5.0

    @staticmethod
    def subtract_constant(x: T) -> T:
        return x - 5.0

    @staticmethod
    def multiply(x: T) -> T:
        return 5.0 * x

    @staticmethod
    def divide(x: T) -> T:
        return x / 5.0

    @staticmethod
    def square(x: T):
        return x * x

    @staticmethod
    def cube(x: T):
        return x * x * x

    @staticmethod
    def sigmoid(x: T) -> T:
        return operators.sigmoid(x)

    @staticmethod
    def inv(x: T):
        return operators.inv(x + 3.5)

    @staticmethod
    def log(x: T) -> T:
        # To assure values are positive for tests.
        return operators.log(x + 1e06)

    @staticmethod
    def relu(x: T) -> T:
        return operators.relu(x + 5.5)

    @staticmethod
    def exp(x: T) -> T:
        return operators.exp(x - 200)

    @staticmethod
    def explog(x: T) -> T:
        return operators.log(x + 1e06) + operators.exp(x - 200)

    @staticmethod
    def add2(x: T, y: T) -> T:
        return x + y

    @staticmethod
    def multiply2(x: T, y: T) -> T:
        return x * y

    @staticmethod
    def divide2(x: T, y: T) -> T:
        return x / (y + 5.5)

    @staticmethod
    def gt2(x: T, y: T) -> T:
        return operators.gt(x + 1.2, y)

    @staticmethod
    def lt2(x: T, y: T) -> T:
        return operators.lt(x + 1.2, y)

    @staticmethod
    def eq2(x: T, y: T) -> T:
        return operators.eq(x, (y + 5.5))

    @staticmethod
    def ge2(x: T, y: T) -> T:
        return operators.ge(x + 1.2, y)

    @staticmethod
    def le2(x: T, y: T) -> T:
        return operators.le(x + 1.2, y)

    @staticmethod
    def summation_reduction(x: List[T]) -> T:
        return summation(x)

    @staticmethod
    def mean_reduction(x: List[T]) -> T:
        return summation(x) / float(len(x))

    @staticmethod
    def mean_full_reduction(x: List[T]) -> T:
        return summation(x) / float(len(x))

    @staticmethod
    def complex(x: T) -> T:
        return (
            operators.log(
                operators.sigmoid(
                    operators.relu(operators.relu(x * 10 + 7) * 6 + 5) * 10
                )
            )
            / 50
        )

    @classmethod
    def _tests(
        cls,
    ) -> Tuple[
        List[Tuple[str, Callable[[T], T]]],
        List[Tuple[str, Callable[[T, T], T]]],
        List[Tuple[str, Callable[[Iterable[T]], T]]],
    ]:
        """
        Collates all tests.
        """
        one_arg_tests = []
        two_arg_tests = []
        reduction_tests = []

        for k in dir(cls):
            if callable(getattr(cls, k)) and not k.startswith("_"):
                base_fn = getattr(cls, k)
                tup = (k, base_fn)
                if k.endswith("2"):
                    two_arg_tests.append(tup)
                elif k.endswith("reduction"):
                    reduction_tests.append(tup)
                else:
                    one_arg_tests.append(tup)

        return one_arg_tests, two_arg_tests, reduction_tests


class MathTestVariable(MathTest[T]):
    @staticmethod
    def square(x: T) -> T:
        return x.square()

    @staticmethod
    def cube(x: T) -> T:
        return x.cube()

    @staticmethod
    def sigmoid(x: T) -> T:
        return x.sigmoid()

    @staticmethod
    def log(x: T) -> T:
        return (x + 1e06).log()

    @staticmethod
    def relu(x: T) -> T:
        return (x + 5.5).relu()

    @staticmethod
    def exp(x: T) -> T:
        return (x - 200.0).exp()

    @staticmethod
    def explog(x: T) -> T:
        return (x + 1e06).log() + (x - 200.0).exp()

    @staticmethod
    def gt2(x: T, y: T) -> T:
        return (x + 1.2) > y

    @staticmethod
    def lt2(x: T, y: T) -> T:
        return (x + 1.2) < y

    @staticmethod
    def eq2(x: T, y: T) -> T:
        return x == (y + 5.5)

    @staticmethod
    def ge2(x: T, y: T) -> T:
        return (x + 1.2) >= y

    @staticmethod
    def le2(x: T, y: T) -> T:
        return (x + 1.2) <= y

    @staticmethod
    def complex(x: T) -> T:
        return (((x * 10 + 7).relu() * 6 + 5).relu() * 10).sigmoid().log() / 50

    @classmethod
    def _comp_testing(
        cls,
    ) -> Tuple[
        List[Tuple[str, Callable[[float], float]], Callable[[T], T]],
        List[Tuple[str, Callable[[float, float], float]], Callable[[T, T], T]],
        List[
            Tuple[str, Callable[[Iterable[float]], float], Callable[[Iterable[T]], T]]
        ],
    ]:
        one_arg, two_arg, red = MathTestOperators._tests()
        one_argv, two_argv, redv = cls._tests()
        one_arg_comp = [(n1, f1, f2) for (n1, f1), (_, f2) in zip(one_arg, one_argv)]
        two_arg_comp = [(n1, f1, f2) for (n1, f1), (_, f2) in zip(two_arg, two_argv)]
        red_comp = [(n1, f1, f2) for (n1, f1), (_, f2) in zip(red, redv)]
        return one_arg_comp, two_arg_comp, red_comp


MathTestOperators = MathTest[float]


MathTestScalars = MathTestVariable[Scalar]


class MathTestTensor(MathTestVariable[Tensor]):
    @staticmethod
    def summation_reduction(x: Tensor) -> Tensor:
        return x.sum(dim=0)

    @staticmethod
    def mean_reduction(x: Tensor) -> Tensor:
        return x.mean(dim=0)

    @staticmethod
    def mean_full_reduction(x: Tensor) -> Tensor:
        return x.mean()
