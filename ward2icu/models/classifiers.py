'''
Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from ward2icu.layers import rnn_layer, Conv1dLayers
from ward2icu.utils import calc_conv_output_length, flatten


class BinaryRNNClassifier(nn.Module):
    def __init__(self, sequence_length, **kwargs):
        """Recursive NN for binary classification with linear 
        time-collapsing layers.

        Layers:
            RNN (with activation, multiple layers):
                input:  (batch_size, sequence_length, input_size)
                output: (batch_size, sequence_length, hidden_size)

            Linear (no activation):
                input: (batch_size, sequence_length, hidden_size)
                output: (batch_size, sequence_length, 1)

        Notes:
            This model adds upon ward2icu.layers.rnn_layer. See docs
            for more information.

        Args:
            sequence_length (int): Number of points in the sequence.
            kwargs: Keyword arguments passed on to ward2icu.layers.rnn_layer
        """
        super(BinaryRNNClassifier, self).__init__()

        # Default hidden_size to input_size.
        hidden_size = kwargs.get('hidden_size', kwargs['input_size'])

        # See ward2icu.layers.rnn_layer for more information.
        self.rnn = rnn_layer(**kwargs)

        # Parameters are shared across time-steps.
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.linear(x).squeeze()


class BinaryCNNClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 input_length,
                 kernel_size,
                 n_layers=1,
                 step_up=1,
                 dropout_prob=0.5,
                 channel_last=True,
                 pool_size=None,
                 **kwargs):
        """ 1D CNN for binary classification. 
        """
        # defaults
        pool_size = pool_size or kernel_size

        self.input_length = input_length
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.channel_last = channel_last
        self.step_up = step_up
        self.pool_size = pool_size
        self.n_layers = n_layers
        super(BinaryCNNClassifier, self).__init__()

        conv = Conv1dLayers(input_size,
                            input_length,
                            kernel_size,
                            n_layers=n_layers,
                            step_up=step_up,
                            dropout_prob=dropout_prob,
                            **kwargs)

        maxpool = nn.MaxPool1d(pool_size)
        flatten = nn.Flatten()
        flatten_output_size = (
            conv.output_sizes[-1]*
            calc_conv_output_length(maxpool, 
                                    conv.output_lengths[-1])
        )

        self.layers = nn.Sequential(conv,
                                    maxpool,
                                    flatten,
                                    nn.Linear(flatten_output_size, flatten_output_size),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(flatten_output_size, 1),
                                    nn.Sigmoid())

    def forward(self, x):
        if self.channel_last:
            x = x.permute(0, 2, 1)
        return self.layers(x).squeeze()
