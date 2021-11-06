#encoding:utf-8

import random
import numpy as np
import glob
import os
import itertools
import time

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

import torchaudio

from module.dataset import *
from module.generator import *
from module.inference import *

#乱数のシードを設定　これにより再現性を確保できる
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#変換したいwavファイルへの、各データへのパスのフォーマット
wav_path = "./dataset/train/domainA/jvs_extracted/ver1/jvs001/VOICEACTRESS100_010.wav"
#scycloneの学習済みモデルへのパス
scyclone_trained_model_path = "./output/scyclone/train/iteration47500/generator_A2B_trained_model_cpu.pth"
#結果を出力するためのディレクトリ
output_dir = "./output/scyclone/inference/"
#使用するデバイス
device = "cuda:0"

#GPUが使用可能かどうか確認
device = torch.device(device if torch.cuda.is_available() else "cpu")
print("device:",device)

#scycloneのGeneratorのインスタンスを生成
netG = Generator()
#学習済みモデルの読み込み
netG.load_state_dict(torch.load(scyclone_trained_model_path))
#ネットワークをデバイスに移動
netG = netG.to(device)

#各wavファイルに対して推論を行う
audio_filepath_list = make_datapath_list(wav_path)
for i, audio_filepath in enumerate(audio_filepath_list, 0):
	#音声ファイルの読み込み
	input_waveform, _ = torchaudio.load(audio_filepath)
	input_spectrogram = torchaudio.transforms.Spectrogram(n_fft=254, hop_length=128)(input_waveform)
	#変換の実行
	input_spectrogram = input_spectrogram.to(device)
	output_spectrogram = inference(input_spectrogram=input_spectrogram, netG=netG)
	#スペクトログラムから負の値を取り除く
	output_spectrogram = F.relu(output_spectrogram)
	#スペクトログラムから波形への変換
	output_spectrogram = output_spectrogram.to("cpu")
	output_waveform = torchaudio.transforms.GriffinLim(n_fft=254, n_iter=256, hop_length=128)(output_spectrogram)
	# print("input:", input_waveform.size(), input_spectrogram.size())
	# print("output:", output_waveform.size(), output_spectrogram.size())
	#音声ファイルを保存
	save_dir = os.path.join(output_dir, f"result_{i}")
	os.makedirs(save_dir, exist_ok=True)#出力用ディレクトリがなければ作る
	torchaudio.save(os.path.join(save_dir, os.path.basename(audio_filepath)), input_waveform, sample_rate=16000)
	torchaudio.save(os.path.join(save_dir, "converted_"+os.path.basename(audio_filepath)), output_waveform, sample_rate=16000)
	print(f'generated {os.path.join(save_dir, "converted_"+os.path.basename(audio_filepath))}')
	#変換前後を比較するためのグラフを出力
	output_comparison_graph(
		save_path=os.path.join(save_dir, "comparison.png"), 
		waveform_source=input_waveform.squeeze(dim=0).to("cpu"), waveform_result=output_waveform.squeeze(dim=0).to("cpu"),      #waveform : torch.size([frame])
		spectrogram_source=input_spectrogram.squeeze(dim=0).to("cpu"), spectrogram_result=output_spectrogram.squeeze(dim=0).to("cpu"), #spectrogram : torch.Size([frequency, frame]
		sampling_rate=16000 #サンプリングレート
	)

