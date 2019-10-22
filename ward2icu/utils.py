import torch
import numpy as np
from math import floor
from ward2icu import make_logger

logger = make_logger(__file__)


def tile(t, length):
    ''' Creates an extra dimension on the tensor t and
    repeats it throughout.'''
    return t.view(-1, 1).repeat(1, length)


def calc_conv_output_length(conv_layer,
                            input_length):
    def _maybe_slice(x):
        return x[0] if isinstance(x, tuple) else x

    l = input_length
    p =_maybe_slice(conv_layer.padding)
    d =_maybe_slice(conv_layer.dilation)
    k =_maybe_slice(conv_layer.kernel_size)
    s =_maybe_slice(conv_layer.stride)
    return floor((l + 2*p - d*(k-1)-1)/s + 1)
                             

def flatten(l):
    return [item for sublist in l for item in sublist]


def numpy_to_cuda(*args):
    return [torch.from_numpy(a).cuda() for a in args]


def n_(tensor):
    return tensor.detach().cpu().numpy()
