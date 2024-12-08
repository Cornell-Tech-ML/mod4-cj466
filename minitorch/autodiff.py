from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_list = list(vals)
    original_value = vals_list[arg]

    vals_list[arg] = original_value + epsilon
    f_plus = f(*vals_list)

    vals_list[arg] = original_value - epsilon
    f_minus = f(*vals_list)

    vals = tuple(vals_list)

    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Unique id of the variable."""
        ...

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        ...

    def is_constant(self) -> bool:
        """True if this variable is constant (no `last_fn`)"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Iterable of the parent variables."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Chain rule for the variable."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    order = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for parent in var.parents:
                if not parent.is_constant():
                    visit(parent)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
    variable: The variable to start backpropagation from.
    deriv: The derivative of the output with respect to the variable.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    queue = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}
    for var in queue:
        derivative = derivatives.get(var.unique_id, 0.0)
        if var.is_leaf():
            var.accumulate_derivative(derivative)
        else:
            for parent_var, parent_derivative in var.chain_rule(derivative):
                if not parent_var.is_constant():
                    derivatives[parent_var.unique_id] = (
                        derivatives.get(parent_var.unique_id, 0.0) + parent_derivative
                    )


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved tensors."""
        return self.saved_values
