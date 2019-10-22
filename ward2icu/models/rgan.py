'''
Reference: https://arxiv.org/abs/1706.02633 
'''

import torch
import torch.nn as nn
from torchgan.models import Generator, Discriminator
from ward2icu.layers import rnn_layer


class RGANGenerator(Generator):
    def __init__(self,
                 sequence_length,
                 output_size,
                 hidden_size=None,
                 noise_size=None,
                 num_layers=1,
                 dropout=0,
                 rnn_nonlinearity='tanh',
                 rnn_type='rnn',
                 input_size=None,
                 last_layer=None,
                 **kwargs):
        """Recursive GAN (Generator) implementation with RNN cells.

        Layers:
            RNN (with activation, multiple layers):
                input:  (batch_size, sequence_length, noise_size)
                output: (batch_size, sequence_length, hidden_size)

            Linear (no activation, weights shared between time steps):
                input:  (batch_size, sequence_length, hidden_size)
                output: (batch_size, sequence_length, output_size)

            last_layer (optional)
                input: (batch_size, sequence_length, output_size)

        Args:
            sequence_length (int): Number of points in the sequence.
                Defined by the real dataset.
            output_size (int): Size of output (usually the last tensor dimension).
                Defined by the real dataset.
            hidden_size (int, optional): Size of RNN output.
                Defaults to output_size.
            noise_size (int, optional): Size of noise used to generate fake data.
                Defaults to output_size.
            num_layers (int, optional): Number of stacked RNNs in rnn.
            dropout (float, optional): Dropout probability for rnn layers.
            rnn_nonlinearity (str, optional): Non-linearity of the RNN. Must be
                either 'tanh' or 'relu'. Only valid if rnn_type == 'rnn'.
            rnn_type (str, optional): Type of RNN layer. Valid values are 'lstm',
                'gru' and 'rnn', the latter being the default.
            input_size (int, optional): Input size of RNN, defaults to noise_size.
            last_layer (Module, optional): Last layer of the discriminator.
        """


        # Defaults
        noise_size = noise_size or output_size
        input_size = input_size or noise_size
        hidden_size = hidden_size or output_size

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_size = noise_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_nonlinearity = rnn_nonlinearity
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.label_type = "none"

        # Set kwargs (might overried above attributes)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Total size of z that will be sampled. Later, in the forward 
        # method, we resize to (batch_size, sequence_length, noise_size).
        # TODO: Any resizing of z is valid as long as the total size 
        #       remains sequence_length*noise_size. How does this affect
        #       the performance of the RNN?
        self.encoding_dims = sequence_length*noise_size

        super(RGANGenerator, self).__init__(self.encoding_dims,
                                            self.label_type)

        # Build RNN layer
        self.rnn = rnn_layer(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             dropout=dropout,
                             rnn_type=rnn_type,
                             nonlinearity=rnn_nonlinearity)
        self.dropout = nn.Dropout(dropout)
        # self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.last_layer = last_layer

        # Initialize all weights.
        # nn.init.xavier_normal_(self.rnn)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, z, reshape=True):
        if reshape:
            z = z.view(-1, self.sequence_length, self.noise_size)
        y, _ = self.rnn(z)
        y = self.dropout(y)
        # y = self.batchnorm(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.linear(y)
        return y if self.last_layer is None else self.last_layer(y)


class RGANDiscriminator(Discriminator):
    def __init__(self,
                 sequence_length,
                 input_size,
                 hidden_size=None,
                 num_layers=1,
                 dropout=0,
                 rnn_nonlinearity='tanh',
                 rnn_type='rnn',
                 last_layer=None,
                 **kwargs):
        """Recursive GAN (Discriminator) implementation with RNN cells.

        Layers:
            RNN (with activation, multiple layers): 
                input:  (batch_size, sequence_length, input_size)
                output: (batch_size, sequence_length, hidden_size)

            Linear (no activation, weights shared between time steps):
                input:  (batch_size, sequence_length, hidden_size)
                output: (batch_size, sequence_length, 1)

            last_layer (optional)
                input: (batch_size, sequence_length, 1)

        Args:
            sequence_length (int): Number of points in the sequence.
            input_size (int): Size of input (usually the last tensor dimension).
            hidden_size (int, optional): Size of hidden layers in rnn. 
                If None, defaults to input_size.
            num_layers (int, optional): Number of stacked RNNs in rnn.
            dropout (float, optional): Dropout probability for rnn layers.
            rnn_nonlinearity (str, optional): Non-linearity of the RNN. Must be
                either 'tanh' or 'relu'. Only valid if rnn_type == 'rnn'.
            rnn_type (str, optional): Type of RNN layer. Valid values are 'lstm',
                'gru' and 'rnn', the latter being the default.
            last_layer (Module, optional): Last layer of the discriminator.
        """

        # TODO: Insert non-linearities between Linear layers.
        # TODO: Add BatchNorm and Dropout as in https://arxiv.org/abs/1905.05928v1

        # Set hidden_size to input_size if not specified
        hidden_size = hidden_size or input_size

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.label_type = "none"

        # Set kwargs (might overried above attributes)
        for key, value in kwargs.items():
            setattr(self, key, value)

        super(RGANDiscriminator, self).__init__(self.input_size,
                                                self.label_type)

        # Build RNN layer
        self.rnn = rnn_layer(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             dropout=dropout,
                             rnn_type=rnn_type,
                             nonlinearity=rnn_nonlinearity)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
        self.last_layer = last_layer

        # Initialize all weights.
        # nn.init.xavier_normal_(self.rnn)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        y, _ = self.rnn(x)
        y = self.dropout(y)
        y = self.linear(y)
        return y if self.last_layer is None else self.last_layer(y)
