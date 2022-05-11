from typing import List, Tuple
import minitorch.operators as operators


class MathTest:

    @staticmethod
    def neg(x: float) -> float:
        return -x

    @staticmethod
    def add_constant(x: float) -> float:
        return x + 5.

    @staticmethod
    def subtract_constant(x: float) -> float:
        return x - 5.

    @staticmethod
    def multiply(x: float) -> float:
        return 5. * x

    @staticmethod
    def divide(x: float) -> float:
        return x / 5.

    @staticmethod
    def sigmoid(x: float) -> float:
        return operators.sigmoid(x)

    @staticmethod
    def log(x: float) -> float:
        return operators.log(abs(x))  # To assure values are positive for tests.

    @staticmethod
    def relu(x: float) -> float:
        return operators.relu(x + 5.5)

    @staticmethod
    def exp(x: float) -> float:
        return operators.exp(x - 200)

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

    @classmethod
    def generate_tests(cls) -> Tuple[List, List, List]:
        """
        Collates all tests.
        """
        one_arg_tests = []
        two_arg_tests = []
        reduction_tests = []
        for k in dir(MathTest):
            if (
                callable(getattr(MathTest, k))
                and
                not (k.startswith("generate") or k.startswith("_"))
            ):
                base_fn = getattr(MathTest, k)
                scalar_fn = getattr(cls, k)
                tup = (k, base_fn, scalar_fn)
                if k.endswith("2"):
                    two_arg_tests.append(tup)
                elif k.endswith("reduction"):
                    reduction_tests.append(tup)
                else:
                    one_arg_tests.append(tup)

        return one_arg_tests, two_arg_tests, reduction_tests

