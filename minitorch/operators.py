"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply `x` by `y`."""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Add `x` and `y`."""
    return x + y


def neg(x: float) -> float:
    """Negate `x`."""
    return -x


def lt(x: float, y: float) -> float:
    """Checks if x is less than y."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if x is equal to y."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close."""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid of a number."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Returns the ReLU of a number."""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Returns the natural log of `x`."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Returns the exponential of `x`."""
    return math.exp(x)


def inv(x: float) -> float:
    """Returns the reciprocal of `x`."""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Returns the derivative of the natural logarithm of `x` multiplied by `y`."""
    return y / (x + EPS)


def inv_back(x: float, y: float) -> float:
    """Returns the derivative of the reciprocal of `x` multiplied by `y`."""
    return -(1.0 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Returns the derivative of the ReLU function of `x` multiplied by `y`."""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a given function to each element of an iterable."""

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Applies a given function to combine elements of two iterables."""

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Applies a given function to an iterable to a single value."""

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in an iterable using map"""
    return map(neg)(ls)


def addLists(x: Iterable[float], y: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two iterables using zipWith."""
    return zipWith(add)(x, y)


def sum(x: Iterable[float]) -> float:
    """Sum all elements in an iterable using reduce."""
    return reduce(add, 0.0)(x)


def prod(x: Iterable[float]) -> float:
    """Calculate the product of all elements in an iterable using reduce."""
    return reduce(mul, 1.0)(x)
