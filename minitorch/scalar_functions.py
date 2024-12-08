from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the input values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs addition of a and b."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Returns the gradient with respect to a and b."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Applies the log function to a."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the gradient with respect to a using log_back."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function f(x, y) = x * y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Applies the multiplication function to a and b."""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Returns the gradient with respect to a and b."""
        (a, b) = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function f(x) = 1 / x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Returns the inverse of a."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the gradient with respect to a."""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function f(x) = -x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Returns the negation of a."""
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the gradient with respect to a."""
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function f(x) = 1 / (1 + exp(-x))"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Applies the sigmoid function to a."""
        sigVal = operators.sigmoid(a)
        ctx.save_for_backward(sigVal)
        return sigVal

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the gradient with respect to a using sigmoid_back."""
        (sigVal,) = ctx.saved_values
        return d_output * sigVal * (1 - sigVal)


class ReLU(ScalarFunction):
    """ReLU function f(x) = max(0, x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Applies the ReLU function to a."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the gradient with respect to a using relu_back."""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function f(x) = exp(x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Applies the exponential function to a."""
        exp_a = operators.exp(a)
        ctx.save_for_backward(exp_a)
        return exp_a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the gradient with respect to a."""
        (exp_a,) = ctx.saved_values
        return d_output * exp_a


class LT(ScalarFunction):
    """Less-than function f(x, y) = x < y ? 1.0 : 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Returns 1.0 if a < b, else 0.0."""
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Gradients for comparison functions are 0."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equality function f(x, y) = x == y ? 1.0 : 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Returns 1.0 if a == b, else 0.0."""
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Gradients for comparison functions are 0."""
        return 0.0, 0.0
