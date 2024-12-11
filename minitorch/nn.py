from typing import Tuple

from minitorch import operators
from minitorch.autodiff import Context
from minitorch.tensor_functions import Function, rand
from .fast_ops import FastOps

from .tensor import Tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # Implement for Task 4.3.
    tile_height = height // kh
    tile_width = width // kw
    input = input.contiguous().view(batch, channel, tile_height, kh, tile_width, kw)
    input = input.permute(0, 1, 2, 4, 3, 5).contiguous()
    input = input.view(batch, channel, tile_height, tile_width, kh * kw)
    return input, tile_height, tile_width


# Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D average pooling to input tensor.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width)
        kernel: Tuple of (kernel_height, kernel_width)

    Returns:
    -------
        Pooled tensor with reduced dimensions

    """
    batch, channel, height, width = input.shape
    input, tile_height, tile_width = tile(input, kernel)
    input = input.mean(4)
    input = input.view(batch, channel, tile_height, tile_width)
    return input


# Implement for Task 4.4.
max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Return a one-hot tensor with 1's where the maximum value is found along dimension dim.

    Args:
    ----
        input: Input tensor
        dim: Dimension to reduce along

    Returns:
    -------
        One-hot tensor with same shape as input

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max should be max reduction"""
        dim_int = int(dim.item())
        ctx.save_for_backward(input, dim)
        return max_reduce(input, dim_int)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for max reduction.

        Args:
        ----
            ctx: Context with saved tensors
            grad_output: Gradient with respect to output

        Returns:
        -------
            Tuple of input gradient and dimension gradient (0.0)

        """
        input, dim = ctx.saved_values
        return grad_output * argmax(input, int(dim.item())), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction along specified dimension.

    Args:
    ----
        input: Input tensor
        dim: Dimension to reduce along

    Returns:
    -------
        Tensor with maximum values along specified dimension

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Apply softmax along specified dimension.

    Args:
    ----
        input: Input tensor
        dim: Dimension to compute softmax over

    Returns:
    -------
        Tensor with softmax applied

    """
    input = input.exp()
    return input / input.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Apply log softmax along specified dimension.

    Args:
    ----
        input: Input tensor
        dim: Dimension to compute log softmax over

    Returns:
    -------
        Tensor with log softmax applied

    """
    return softmax(input, dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D max pooling to input tensor.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width)
        kernel: Tuple of (kernel_height, kernel_width)

    Returns:
    -------
        Pooled tensor with reduced dimensions

    """
    batch, channel, height, width = input.shape
    input, tile_height, tile_width = tile(input, kernel)
    max_input = max(input, 4)
    return max_input.view(batch, channel, tile_height, tile_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to input tensor.

    Args:
    ----
        input: Input tensor
        rate: Dropout rate (probability of setting values to zero)
        ignore: If True, disable dropout

    Returns:
    -------
        Tensor with dropout applied

    """
    if not ignore:
        random = rand(input.shape) > rate
        input = input * random
    return input
