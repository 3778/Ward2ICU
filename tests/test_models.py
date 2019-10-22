import torch
import numpy as np
import pytest
from torchgan.losses import MinimaxDiscriminatorLoss
from sybric.models import (RGANGenerator, 
                           RGANDiscriminator,
                           RCGANGenerator, 
                           RCGANDiscriminator,
                           BinaryRNNClassifier,
                           BinaryCNNClassifier,
                           CNNCGANGenerator,
                           CNNCGANDiscriminator,
                           FCCMLPGANGenerator,
                           FCCMLPGANDiscriminator)

np.random.seed(3778)
torch.manual_seed(3778)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE='cpu'

@pytest.fixture
def RGAN_setup():
    batch_size = 3
    seq_length = 10

    # Generator
    gen_opt = dict(output_size=5,
                   num_layers=1,
                   dropout=0,
                   noise_size=100,
                   sequence_length=seq_length)

    # Discriminator
    dsc_opt = dict(input_size=gen_opt['output_size'],
                   sequence_length=seq_length,
                   hidden_size=None,
                   num_layers=1,
                   dropout=0)

    gen = RGANGenerator(**gen_opt)
    dsc = RGANDiscriminator(**dsc_opt)
    return gen, dsc, batch_size, seq_length


def test_RGAN_forward(RGAN_setup):
    gen, dsc, batch_size, seq_length = RGAN_setup
    z, = gen.sampler(batch_size, DEVICE)
    assert tuple(z.shape) == (batch_size, seq_length*gen.noise_size)

    x = gen.forward(z)
    assert tuple(x.shape) == (batch_size, seq_length, gen.output_size)

    y = dsc.forward(x)
    assert tuple(y.shape) == (batch_size, seq_length, 1)



def test_RCGAN():
    batch_size = 3
    seq_length = 20
    label_emb_size = None
    prob_classes = [0, 0, 0, 0, 1]
    num_classes = len(prob_classes)

    # Generator
    gen_opt = dict(num_classes=num_classes,
                   prob_classes=prob_classes,
                   label_embedding_size=label_emb_size,
                   output_size=5,
                   num_layers=1,
                   dropout=0,
                   noise_size=10,
                   hidden_size=100,
                   sequence_length=seq_length)

    # Discriminator
    dsc_opt = dict(input_size=gen_opt['output_size'],
                   sequence_length=seq_length,
                   num_classes=num_classes,
                   label_embedding_size=label_emb_size,
                   num_layers=1,
                   hidden_size=100,
                   dropout=0)

    gen = RCGANGenerator(**gen_opt)
    dsc = RCGANDiscriminator(**dsc_opt)

    z, y = gen.sampler(batch_size, DEVICE)
    assert tuple(z.shape) == (batch_size, seq_length*gen.noise_size)
    assert tuple(y.shape) == (batch_size,)

    x = gen.forward(z, y)
    assert tuple(x.shape) == (batch_size, seq_length, gen.output_size)

    p = dsc.forward(x, y)
    assert tuple(p.shape) == (batch_size, seq_length, 1)

    # assert prob_classes
    assert (y == 4).all()


def test_BinaryRNNClassifier():
    seq_length = 10
    batch_size = 8
    hidden_size = 20
    input_size = 2

    model = BinaryRNNClassifier(sequence_length=seq_length,
                                input_size=input_size,
                                hidden_size=hidden_size)

    x = torch.randn(batch_size, seq_length, input_size)
    p = model.forward(x)
    assert tuple(p.shape) == (batch_size, seq_length)


def test_BinaryCNNClassifier():
    input_size = 2
    input_length = 10
    batch_size = 8

    model = BinaryCNNClassifier(input_size,
                                input_length,
                                kernel_size=3)

    x = torch.randn(batch_size, input_length, input_size)
    assert tuple(x.shape) == (batch_size, input_length, input_size)

    p = model.forward(x)
    assert tuple(p.shape) == (batch_size,)


def test_FCCMLPGAN():
    sequence_length = 20
    sequence_size = 5
    hidden_size = 10
    prob_classes = [0, 0, 0, 0, 1]
    num_classes = len(prob_classes)
    batch_size = 8

    # Generator
    gen_opt = dict(sequence_length=sequence_length,
                   sequence_size=sequence_size,
                   num_classes=num_classes,
                   hidden_size=hidden_size,
                   prob_classes=prob_classes)

    # Discriminator
    dsc_opt = dict(sequence_length=sequence_length,
                   num_classes=num_classes,
                   sequence_size=gen_opt['sequence_size'],
                   hidden_size=hidden_size)

    gen = FCCMLPGANGenerator(**gen_opt)
    dsc = FCCMLPGANDiscriminator(**dsc_opt)

    z, y = gen.sampler(batch_size, DEVICE)
    assert tuple(z.shape) == (batch_size, sequence_length*gen.noise_size)
    assert tuple(y.shape) == (batch_size,)

    x = gen.forward(z, y)
    assert tuple(x.shape) == (batch_size, sequence_length, sequence_size)

    p = dsc.forward(x, y)
    assert tuple(p.shape) == (batch_size, sequence_length)

    # assert prob_classes
    assert (y == 4).all()
