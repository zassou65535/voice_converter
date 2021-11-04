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

class ResidualBlock_G(nn.Module):
	def __init__(self, in_and_out_channel):
		super().__init__()
		
		self.conv_block = nn.Sequential(
			nn.Conv1d(in_and_out_channel, in_and_out_channel, kernel_size=5, stride=1, padding=2),
			nn.LeakyReLU(0.01),
			nn.Conv1d(in_and_out_channel, in_and_out_channel, kernel_size=5, stride=1, padding=2),
		)

	def forward(self, x):
		return x + self.conv_block(x)


class Generator(nn.Module):
	def __init__(self):
		super().__init__()

		self.model = nn.Sequential(
			nn.Conv1d(128, 256, kernel_size=1, stride=1),
			nn.LeakyReLU(0.01),
			ResidualBlock_G(in_and_out_channel=256),
			ResidualBlock_G(in_and_out_channel=256),
			ResidualBlock_G(in_and_out_channel=256),
			ResidualBlock_G(in_and_out_channel=256),
			ResidualBlock_G(in_and_out_channel=256),
			ResidualBlock_G(in_and_out_channel=256),
			ResidualBlock_G(in_and_out_channel=256),
			nn.Conv1d(256, 128, kernel_size=1, stride=1),
			nn.LeakyReLU(0.01)
		)

	def forward(self, x):
		return self.model(x)
