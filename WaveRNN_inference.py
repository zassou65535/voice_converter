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
from module.vocoder import *

#乱数のシードを設定
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#対象とするwavファイルへのパスのフォーマット
audio_path = "./dataset/train/domainA/jvs_extracted/ver2/jvs004/VOICEACTRESS100_010.wav"
#WaveRNNの学習済みVocoderへのパス
wavernn_trained_model_path = "./output/wavernn/train/iteration90000/vocoder_trained_model_cpu.pth"
#結果を出力するためのディレクトリ
output_dir = "./output/wavernn/inference/"
#使用するデバイス
device = "cuda:2"

#出力用ディレクトリがなければ作る
os.makedirs(output_dir, exist_ok=True)

#GPUが使用可能かどうか確認
device = torch.device(device if torch.cuda.is_available() else "cpu")
print("device:",device)

#WaveRNNのVocoderのインスタンスを生成
vocoder = WaveRNNVocoder()
#学習済みモデルの読み込み
vocoder.load_state_dict(torch.load(wavernn_trained_model_path))
#ネットワークをデバイスに移動
vocoder = vocoder.to(device)

#wavファイルをロード、スペクトログラムを生成
input_waveform, _ = torchaudio.load(audio_path)
input_spectrogram = torchaudio.transforms.Spectrogram(n_fft=254, hop_length=128)(input_waveform)
#vocoderによる推論を実行、スペクトログラムから波形を生成する
vocoder.eval()
output_waveform_by_vocoder = vocoder.generate(input_spectrogram.transpose(1, 2).to(device))[None, ...]
vocoder.train()
#GriffinLimによってスペクトログラムから波形を生成する(vocoderによる生成結果との比較用)
output_waveform_by_griffinlim = torchaudio.transforms.GriffinLim(n_fft=254, n_iter=256, hop_length=128)(input_spectrogram)

#結果を保存する
torchaudio.save(os.path.join(output_dir, "input_audio.wav"), input_waveform, sample_rate=16000)
torchaudio.save(os.path.join(output_dir, "output_audio_by_vocoder.wav"), output_waveform_by_vocoder, sample_rate=16000)
#比較用として、GriffinLimによって生成した波形も出力する
torchaudio.save(os.path.join(output_dir, "output_audio_by_griffinlim.wav"), output_waveform_by_griffinlim, sample_rate=16000)

#音声を、波形とスペクトログラム2つの観点で比較するためのグラフを出力する
waveform_list = [
	(input_waveform.squeeze(dim=0).to("cpu"), "input_waveform"),
	(output_waveform_by_vocoder.squeeze(dim=0).to("cpu"), "output_waveform_by_vocoder"),
	(output_waveform_by_griffinlim.squeeze(dim=0).to("cpu"), "output_waveform_by_griffinlim"),
]
spectrogram_list = [
	(input_spectrogram.squeeze(dim=0).to("cpu"), "input_spectrogram"),
]
output_comparison_graph(
		save_path = os.path.join(output_dir, "comparison.png"),
		waveform_list=waveform_list,     #waveform_list : (torch.size([frame]), graph_title)を要素に持つlist
		spectrogram_list=spectrogram_list, #spectrogram_list : (torch.Size([frequency, frame]), graph_title)を要素に持つlist
		sampling_rate=16000, #サンプリングレート
	)

