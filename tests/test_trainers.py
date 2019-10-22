import pytest
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from sybric.trainers import (BinaryClassificationTrainer,
                             MinMaxBinaryCGANTrainer)
from sybric.samplers import IdentitySampler
from sybric.models import RCGANGenerator, RCGANDiscriminator
from itertools import chain

@pytest.fixture
def binaryclass_trainer():
    batch_size = 8
    input_size = 3
    num_classes = 2

    X_train = torch.randn(batch_size, input_size, device='cpu').float()
    y_train = torch.randint(0, num_classes, (batch_size,), device='cpu').float()

    X_test = torch.randn(batch_size, input_size, device='cpu').float()
    y_test = torch.randint(0, num_classes, (batch_size,), device='cpu').float()

    sampler_train = IdentitySampler(X_train, y_train)
    sampler_test = IdentitySampler(X_test, y_test)

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = nn.Linear(input_size, 1)
        def forward(self, x):
            return self.linear(x).squeeze()

    model = Model().cpu()
    optimizer = SGD(model.parameters(), lr=100)
    binaryclass_trainer = BinaryClassificationTrainer(optimizer=optimizer,
                                          sampler_train=sampler_train,
                                          sampler_test=sampler_test,
                                          model=model)
    return binaryclass_trainer

@pytest.fixture
def binaryclass_tiled_trainer():
    batch_size = 8
    input_size = 3
    num_classes = 2

    X_train = torch.randn(batch_size, input_size, device='cpu').float()
    y_train = torch.randint(0, num_classes, (batch_size,), device='cpu').float()

    X_test = torch.randn(batch_size, input_size, device='cpu').float()
    y_test = torch.randint(0, num_classes, (batch_size,), device='cpu').float()

    sampler_train = IdentitySampler(X_train, y_train, tile=True)
    sampler_test = IdentitySampler(X_test, y_test, tile=True)

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = nn.Linear(input_size, 1)
        def forward(self, x):
            return self.linear(x).squeeze()

    model = Model().cpu()
    optimizer = SGD(model.parameters(), lr=100)
    binaryclass_trainer_tiled = BinaryClassificationTrainer(optimizer=optimizer,
                                                            sampler_train=sampler_train,
                                                            sampler_test=sampler_test,
                                                            model=model)
    return binaryclass_trainer_tiled


@pytest.fixture
def minmaxgan_trainer():
    data_size = 10
    input_length = 20
    num_classes = 2
    noise_size = 10
    output_size = 5
    input_size = output_size
    encoding_dims = input_length*input_size

    X = torch.randn(data_size, input_length, output_size)
    y = torch.randint(0, num_classes, (data_size,))
    sampler = IdentitySampler(X, y)

    generator = RCGANGenerator(input_length, output_size,
                               num_classes, noise_size)
    discriminator = RCGANDiscriminator(input_length, num_classes, input_size)
    optimizer_gen = SGD(generator.parameters(), lr=0.1)
    optimizer_dsc = SGD(discriminator.parameters(), lr=0.1)
    trainer = MinMaxBinaryCGANTrainer(generator,
                                      discriminator,
                                      optimizer_gen,
                                      optimizer_dsc,
                                      sampler)
    return trainer

def test_MinMaxGANTrainer_train(minmaxgan_trainer):
    def parameters():
        yield from minmaxgan_trainer.generator.parameters()
        yield from minmaxgan_trainer.discriminator.parameters()

    old_params = [p.detach().clone() for p in parameters()]
    minmaxgan_trainer.train(epochs=1)
    new_params = [p.detach().clone() for p in parameters()]

    # assert parameters are being trained
    assert all([(old != new).all() 
                for old, new in zip(old_params, new_params)])


def test_MinMaxGANTrainer_calc_dsc_loss(minmaxgan_trainer):
    dsc_logits_real = torch.Tensor([[[0],
                                     [0],
                                     [0]],
                            
                                    [[1],
                                     [1],
                                     [0]]])

    dsc_logits_fake = torch.Tensor([[[0],
                                     [0],
                                     [0]],
                            
                                    [[1],
                                     [1],
                                     [0]]])

    loss_real = minmaxgan_trainer.calc_dsc_real_loss(dsc_logits_real)
    loss_fake = minmaxgan_trainer.calc_dsc_fake_loss(dsc_logits_fake)
    loss = loss_real + loss_fake
    e =-0.5*(np.log(1 - torch.sigmoid(dsc_logits_real)).mean() +
             np.log(torch.sigmoid(dsc_logits_fake)).mean())
    assert np.isclose(e, loss.item())


def test_MinMaxGANTrainer_calc_gen_loss(minmaxgan_trainer):
    dsc_logits = torch.Tensor([[[0],
                                [0],
                                [0]],
                            
                               [[1],
                                [1],
                                [0]]])

    loss = minmaxgan_trainer.calc_gen_loss(dsc_logits)
    e = -np.log(torch.sigmoid(dsc_logits)).mean()
    assert np.isclose(e, loss.item())


def test_MinMaxGANTrainer_calculate_metrics(minmaxgan_trainer):
    loss_gen = torch.Tensor([0])
    loss_dsc = torch.Tensor([1])
    metrics = minmaxgan_trainer.calculate_metrics(loss_gen, loss_dsc)
    assert metrics['generator_loss'] == 0
    assert metrics['discriminator_loss'] == 1
    assert np.isnan(metrics['generator_mean_abs_grad'])
    assert np.isnan(metrics['discriminator_mean_abs_grad'])


def test_BinaryClassificationTrainer_train(binaryclass_trainer):
    old_params = [p.detach().clone() 
                  for p in binaryclass_trainer.model.parameters()]
    binaryclass_trainer.train(epochs=1)
    new_params = [p.detach().clone() 
                  for p in binaryclass_trainer.model.parameters()]

    # assert parameters are being trained
    assert all([(old != new).all() for old, new in zip(old_params, new_params)])


def test_BinaryClassificationTrainer_calculate_metrics(binaryclass_trainer):
    logits = torch.Tensor([-9, -8, 8, 9, -1, 8])
    probs = torch.sigmoid(logits)
    y_true = torch.Tensor([ 0,  0, 0, 0,  1, 1])
    y_pred = torch.Tensor([ 0,  0, 1, 1,  0, 1])

    assert (y_pred == probs.round()).all()

    r = binaryclass_trainer.calculate_metrics(y_true, y_pred,
                                              logits, probs)

    loss = -np.log(probs[y_true==1]).sum() - np.log(1 - probs[y_true==0]).sum()
    loss /= probs.shape[0]
    assert np.isclose(r['_accuracy'], 3/6)
    assert np.isclose(r['_balanced_accuracy'], (2/4)*(2/6) + (1/2)*(4/6)) # 0.5
    assert np.isclose(r['_accuracy_0'], 2/4)
    assert np.isclose(r['_accuracy_1'], 1/2)
    assert np.isclose(r['_loss'], loss)
    assert np.isclose(r['_loss_0'], -np.log(1 - probs[y_true==0]).mean())
    assert np.isclose(r['_loss_1'], -np.log(probs[y_true==1]).mean())
    assert np.isclose(r['_matthews'], 0)


def test_BinaryClassificationTrainer_calculate_metrics_tiled(binaryclass_tiled_trainer):
    logits = torch.Tensor([[-9, -8, 8, 9, -1, 8],
                           [-9, -8, 8, 9, -1, 8]]).T
    probs = torch.sigmoid(logits)
    y_true = torch.Tensor([[ 0,  0, 0, 0,  1, 1],
                           [ 0,  0, 0, 0,  1, 1]]).T
    y_pred = torch.Tensor([[ 0,  0, 1, 1,  0, 1],
                           [ 0,  0, 1, 1,  0, 1]]).T

    assert (y_pred == probs.round()).all()

    r = binaryclass_tiled_trainer.calculate_metrics(y_true, y_pred,
                                                    logits, probs)

    loss = 0.5*(-np.log(probs[y_true==1]).sum() - np.log(1 - probs[y_true==0]).sum())
    loss /= probs.shape[0]
    assert np.isclose(r['_accuracy'], 3/6)
    assert np.isclose(r['_balanced_accuracy'], (2/4)*(2/6) + (1/2)*(4/6)) # 0.5
    assert np.isclose(r['_accuracy_0'], 2/4)
    assert np.isclose(r['_accuracy_1'], 1/2)
    assert np.isclose(r['_loss'], loss)
    assert np.isclose(r['_loss_0'], -np.log(1 - probs[y_true==0]).mean())
    assert np.isclose(r['_loss_1'], -np.log(probs[y_true==1]).mean())
    assert np.isclose(r['_matthews'], 0)
