from typing import Tuple

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
