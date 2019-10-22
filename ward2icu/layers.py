import torch.nn as nn
import torch
from ward2icu.utils import calc_conv_output_length, tile


def rnn_layer(input_size, 
              hidden_size=None, 
              num_layers=1, 
              dropout=0.5, 
              rnn_type='rnn',
              nonlinearity='relu'):

    # Set hidden_size to input_size if not specified
    hidden_size = hidden_size or input_size

    rnn_types = {
        'rnn': nn.RNN,
        'lstm': nn.LSTM,
        'gru': nn.GRU
    }

    rnn_kwargs = dict(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout
    )

    if rnn_type == 'rnn':
        rnn_kwargs['nonlinearity'] = nonlinearity

    return rnn_types[rnn_type](**rnn_kwargs)


class Conv1dLayers(nn.Module):
    def __init__(self,
                 input_size,
                 input_length,
                 kernel_size,
                 dropout_prob=0.5,
                 n_layers=1,
                 step_up=1,
                 **kwargs):
        self.input_size = input_size
        self.input_length = input_length
        self.kernel_size = kernel_size
        self.dropout_prob = dropout_prob
        self.n_layers = n_layers
        self.step_up = step_up
        super(Conv1dLayers, self).__init__()

        self.output_sizes = list()
        self.output_lengths = list()

        in_channels = input_size
        out_channels = in_channels
        output_length = input_length
        layers = list()
        for depth in range(n_layers):
            layers += [nn.Conv1d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 **kwargs),
                       nn.Dropout(dropout_prob),
                       nn.LeakyReLU(0.2)]
            output_length = calc_conv_output_length(layers[-3],
                                                    output_length)
            self.output_sizes += [out_channels]
            self.output_lengths += [output_length]

            # next layer
            in_channels = out_channels
            out_channels = step_up*in_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Permutation(nn.Module):
    def __init__(self, *dims):
        self.dims = dims
        super(Permutation, self).__init__()

    def forward(self, x):
        return x.permute(*self.dims)


class View(nn.Module):
    def __init__(self, *dims):
        self.dims = dims
        super(View, self).__init__()

    def forward(self, x):
        return x.view(*self.dims)


class AppendEmbedding(nn.Module):
    def __init__(self, embedding_layer, dim=-1, tile=True):

        super(AppendEmbedding, self).__init__()
        self.embedding_layer = embedding_layer
        self.dim = dim
        self.tile = tile

    def forward(self, x):
        y = self.labels_pointer
        if self.tile:
            y = tile(y, x.size(1))
        y_emb = self.embedding_layer(y.type(torch.LongTensor).to(y.device))
        return torch.cat((x, y_emb), dim=self.dim)
