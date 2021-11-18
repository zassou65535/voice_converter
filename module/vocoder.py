#encoding:utf-8

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models,transforms
import torchvision.utils as vutils
import torch.nn.init as init
from torch.autograd import Function
import torch.nn.functional as F

from .dataset import *

#参考 : https://github.com/anandaswarup/waveRNN
def _init_GRUCell(gru_layer):
    """Instantiate GRUCell with the same paramters as the GRU layer
    """
    gru_cell = nn.GRUCell(gru_layer.input_size, gru_layer.hidden_size)
    gru_cell.weight_hh.data = gru_layer.weight_hh_l0.data
    gru_cell.weight_ih.data = gru_layer.weight_ih_l0.data
    gru_cell.bias_hh.data = gru_layer.bias_hh_l0.data
    gru_cell.bias_ih.data = gru_layer.bias_ih_l0.data
    return gru_cell


class Vocoder(nn.Module):
    def __init__(
                self,
                n_spectrograms = 128,
                hop_length = 128,
                num_bits = 10,
                audio_embedding_dim = 256,
                conditioning_rnn_size = 128,
                rnn_size = 896,
                fc_size = 1024
            ):
        super().__init__()

        self.n_spectrograms = n_spectrograms
        self.hop_length = hop_length
        self.num_bits = num_bits
        self.audio_embedding_dim = audio_embedding_dim
        self.conditioning_rnn_size = conditioning_rnn_size
        self.rnn_size = rnn_size
        self.fc_size = fc_size

        # Conditioning network
        self.conditioning_network = nn.GRU(input_size=n_spectrograms,
                                           hidden_size=conditioning_rnn_size,
                                           num_layers=2,
                                           batch_first=True,
                                           bidirectional=True)

        # Quantized audio embedding
        self.quantized_audio_embedding = nn.Embedding(
            num_embeddings=2**num_bits, embedding_dim=audio_embedding_dim)

        # Autoregressive RNN
        self.rnn = nn.GRU(input_size=audio_embedding_dim +
                          2 * conditioning_rnn_size,
                          hidden_size=rnn_size,
                          batch_first=True)

        # Affine layers
        self.linear_layer = nn.Linear(in_features=rnn_size,
                                      out_features=fc_size)

        self.output_layer = nn.Linear(in_features=fc_size,
                                      out_features=2**num_bits)

    def forward(self, qwavs, spectrograms):
        # Conditioning network
        spectrograms, _ = self.conditioning_network(spectrograms)

        # Upsampling
        spectrograms = F.interpolate(spectrograms.transpose(1, 2), scale_factor=self.hop_length)
        spectrograms = spectrograms.transpose(1, 2)

        # Quantized audio embedding
        embedded_qwavs = self.quantized_audio_embedding(qwavs)

        # Autoregressive RNN
        x, _ = self.rnn(torch.cat((embedded_qwavs, spectrograms), dim=2))

        x = self.output_layer(F.relu(self.linear_layer(x)))

        return x

    def generate(self, spectrogram):
        """Inference mode (Generates an audio waveform from a spectrogram)
        """
        wav = []
        gru_cell = _init_GRUCell(self.rnn)

        # Conditioning network
        spectrogram, _ = self.conditioning_network(spectrogram)

        # Upsampling
        spectrogram = F.interpolate(spectrogram.transpose(1, 2), scale_factor=self.hop_length)
        spectrogram = spectrogram.transpose(1, 2)

        h = torch.zeros(spectrogram.size(0), self.rnn_size, device=spectrogram.device)
        x = torch.zeros(spectrogram.size(0), device=spectrogram.device, dtype=torch.long)
        x = x.fill_(2**(self.num_bits - 1))

        for spectrogram_frame in torch.unbind(spectrogram, dim=1):
            # Audio embedding
            x = self.quantized_audio_embedding(x)

            # Autoregressive GRU Cell
            h = gru_cell(torch.cat((x, spectrogram_frame), dim=1), h)

            x = F.relu(self.linear_layer(h))
            logits = self.output_layer(x)

            # Apply softmax over the logits and generate a distribution
            posterior = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(posterior)

            # Sample from the distribution to generate output
            x = dist.sample()
            wav.append(x.item())
            
        wav = torch.FloatTensor(wav)
        wav = mu_law_expansion(waveform_quantized=wav, bit=10)

        return wav