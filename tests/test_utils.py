from torch import Tensor
from sybric.utils import tile


def test_tile():
    y = Tensor([0, 1, 2])
    y_tiled = tile(y, 3)
    expected = Tensor([[0, 0, 0],
                       [1, 1, 1], 
                       [2, 2, 2]])
    assert (y_tiled == expected).all()
