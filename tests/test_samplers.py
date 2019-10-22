import torch
from sybric.samplers import BinaryBalancedSampler, IdentitySampler


def test_BinaryBalancedSampler():
    X = torch.eye(5)
    y = torch.Tensor([0, 0, 0, 1, 1])
    sampler = BinaryBalancedSampler(X, y)
    for _ in range(100):
        X_s, y_s = sampler.sample()

        assert (y_s == torch.Tensor([0, 0, 1, 1])).all()
        assert X_s.shape == (4, 5)
        assert any((X_s[0] == X[i]).all() for i in [0, 1, 2, 3])
        assert (X_s[-1] == X[-1]).all()

def test_BinaryBalancedSampler_batch():
    X = torch.eye(5)
    y = torch.Tensor([0, 0, 0, 1, 1])
    sampler = BinaryBalancedSampler(X, y, batch_size=2)
    for _ in range(100):
        X_s, y_s = sampler.sample()

        assert (y_s == torch.Tensor([0, 1])).all()
        assert X_s.shape == (2, 5)
        assert any((X_s[0] == X[i]).all() for i in [0, 1, 2, 3])
        assert any((X_s[-1] == X[i]).all() for i in [-1, -2])

def test_IdentitySampler():
    X = torch.ones(4, 2, 6)
    y = torch.Tensor([0, 0, 1, 1])
    sampler = IdentitySampler(X, y, tile=True)
    X_s, y_s = sampler.sample()

    assert X_s.shape == (4, 2, 6)
    assert y_s.shape == (4, 2)
    assert (y_s == torch.Tensor([[0, 0],
                                 [0, 0],
                                 [1, 1],
                                 [1, 1]])).all()
