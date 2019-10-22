'''
Reference: https://arxiv.org/abs/1806.01875
'''

import torch
import torch.nn as nn
from torch.nn import (Linear, 
                      Conv1d,
                      MaxPool1d,
                      AvgPool1d,
                      Upsample,
                      ReplicationPad1d,
                      LeakyReLU,
                      Flatten,
                      Dropout)
from torchgan.models import Generator, Discriminator
from ward2icu.utils import tile
from ward2icu.layers import Conv1dLayers, View, AppendEmbedding
from ward2icu import set_seeds


class CNNCGANGenerator(Generator):
    def __init__(self,
                 output_size,
                 dropout=0,
                 noise_size=20,
                 channel_last=True,
                 hidden_size=50,
                 **kwargs):
        # defaults
        self.initial_length = 5
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.encoding_dims = noise_size
        self.num_classes = 2
        self.label_embedding_size = self.num_classes
        self.prob_classes  = torch.ones(self.num_classes)
        self.dropout = dropout
        self.label_type = 'generated'
        self.channel_last = channel_last
        self._labels = None

        # Set kwargs (might overried above attributes)
        for key, value in kwargs.items():
            setattr(self, key, value)

        super(CNNCGANGenerator, self).__init__(self.encoding_dims,
                                               self.label_type)
        self.label_embeddings = nn.Embedding(self.num_classes,
                                             self.label_embedding_size)

        Conv1d_ = lambda k: Conv1d(self.hidden_size, self.hidden_size, k)
        ## Build CNN layer
        # (batch_size, channels, sequence length)

        layers_input_size = self.encoding_dims*(1 + self.label_embedding_size)
        self.layers = nn.Sequential(
            Linear(layers_input_size, self.initial_length*self.hidden_size),
            View(-1, self.hidden_size, self.initial_length),
            LeakyReLU(0.2),
            Dropout(dropout),
            # output size: (-1, 50, 5)

            Upsample(scale_factor=2, mode='linear', align_corners=True),
            # output size: (-1, 50, 10)

            Conv1d_(3),
            ReplicationPad1d(1),
            LeakyReLU(0.2),
            Conv1d_(3),
            ReplicationPad1d(1),
            LeakyReLU(0.2),
            Dropout(dropout),
            # output size: (-1, 50, 10)

            Upsample(scale_factor=2, mode='linear', align_corners=True),
            # output size: (-1, 50, 20)

            Conv1d_(3),
            ReplicationPad1d(1),
            LeakyReLU(0.2),
            Conv1d_(3),
            ReplicationPad1d(1),
            LeakyReLU(0.2),
            Dropout(dropout),
            # output size: (-1, 50, 20)

            Conv1d(self.hidden_size, self.output_size, 1)
            # output size: (-1, 5, 20)
        )
      
        # Initialize all weights.
        self._weight_initializer()

    def forward(self, z, y):
        y_tiled = tile(y, z.size(-1))
        y_emb = self.label_embeddings(y_tiled
                                      .type(torch.LongTensor)
                                      .to(y.device))
        z = torch.cat((z.unsqueeze(-1), y_emb), dim=-1)
        z = z.flatten(1)
        x = self.layers(z)
        return x.permute(0, 2, 1) if self.channel_last else x

    def sampler(self, sample_size, device='cpu'):
        return [
            torch.randn(sample_size, self.encoding_dims, device=device),
            torch.multinomial(self.prob_classes, sample_size,
                              replacement=True).to(device)
        ]


class CNNCGANDiscriminator(Generator):
    def __init__(self,
                 input_size,
                 channel_last=True,
                 hidden_size=50,
                 **kwargs):

        # Defaults
        self.input_length = 20
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = 2
        self.label_embedding_size = self.num_classes
        self.prob_classes  = torch.ones(self.num_classes)
        self.dropout = 0.5
        self.label_type = 'required'
        self.channel_last = channel_last


        # Set kwargs (might overried above attributes)
        for key, value in kwargs.items():
            setattr(self, key, value)

        super(CNNCGANDiscriminator, self).__init__(self.input_size,
                                                   self.label_type)

        # Build CNN layer
        self.label_embeddings = nn.Embedding(self.num_classes,
                                             self.label_embedding_size)

        Conv1d_ = lambda k: Conv1d(hidden_size, hidden_size, k)
        layers_input_size = self.input_size + self.label_embedding_size
        layers_output_size = 5*hidden_size
        ## Build CNN layer
        self.layers = nn.Sequential(
            Conv1d(layers_input_size, hidden_size, 1),
            LeakyReLU(0.2),
            # output size: (-1, 50, 20)

            Conv1d_(3),
            ReplicationPad1d(1),
            LeakyReLU(0.2),
            Conv1d_(3),
            ReplicationPad1d(1),
            LeakyReLU(0.2),
            AvgPool1d(2, 2),
            # output size: (-1, 50, 10)

            Conv1d_(3),
            ReplicationPad1d(1),
            LeakyReLU(0.2),
            Conv1d_(3),
            ReplicationPad1d(1),
            LeakyReLU(0.2),
            AvgPool1d(2, 2),
            # output size: (-1, 50, 5)

            Flatten(),
            Linear(layers_output_size, 1)
            # output size: (-1, 1)
        )
      
        # Initialize all weights.
        self._weight_initializer()

    def forward(self, x, y):
        if self.channel_last:
            x = x.permute(0, 2, 1)
        y_tiled = tile(y, x.size(-1))
        y_emb = self.label_embeddings(y_tiled
                                      .type(torch.LongTensor)
                                      .to(y.device))
        y_emb = y_emb.permute(0, 2, 1)

        x = torch.cat((x, y_emb), dim=1)
        x = self.layers(x)
        return x.squeeze()
