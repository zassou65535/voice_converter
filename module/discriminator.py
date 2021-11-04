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

class ResidualBlock_D(nn.Module):
	def __init__(self, in_and_out_channel):
		super().__init__()

		self.conv_blocks = nn.Sequential(
			nn.utils.spectral_norm(nn.Conv1d(in_and_out_channel, in_and_out_channel, kernel_size=5, stride=1, padding=2)),
			nn.LeakyReLU(0.2),
			nn.utils.spectral_norm(nn.Conv1d(in_and_out_channel, in_and_out_channel, kernel_size=5, stride=1, padding=2)),
		)

	def forward(self, x):
		return x + self.conv_blocks(x)


class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()

		self.model = nn.Sequential(
			nn.utils.spectral_norm(nn.Conv1d(128, 256, kernel_size=1, stride=1)),
			nn.LeakyReLU(0.2),
			ResidualBlock_D(in_and_out_channel=256),
			ResidualBlock_D(in_and_out_channel=256),
			ResidualBlock_D(in_and_out_channel=256),
			ResidualBlock_D(in_and_out_channel=256),
			ResidualBlock_D(in_and_out_channel=256),
			ResidualBlock_D(in_and_out_channel=256),
			nn.utils.spectral_norm(nn.Conv1d(256, 1, kernel_size=1, stride=1)),
			nn.LeakyReLU(0.2),
			nn.AdaptiveAvgPool1d(1)
		)

	def forward(self, x):
		#入力にGaussianノイズN(0, 0.01)を加算したものをモデルへの入力とする
		return self.model(x + torch.randn(x.size(), device=x.device)*0.01)


