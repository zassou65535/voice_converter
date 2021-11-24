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
from module.vocoder import *

#乱数のシードを設定
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#変換したいwavファイルへのパス
audio_path = "./dataset/train/domainA/jvs_extracted/ver1/jvs001/VOICEACTRESS100_010.wav"
#Scycloneの学習済みGeneratorへのパス
scyclone_trained_model_path = "./output/scyclone/train/iteration300000/generator_A2B_trained_model_cpu.pth"
#学習済みVocoderへのパス
vocoder_trained_model_path = "./output/vocoder/train/iteration80000/vocoder_trained_model_cpu.pth"
#結果を出力するためのディレクトリ
output_dir = "./output/scyclone/inference/"
#使用するデバイス
device = "cuda:0"
#スペクトログラムを何フレームごとにモデルを用いて変換するか
unit_frame=160
#変換後のスペクトログラムから、中央何フレーム分を切り出すか
cutout_frame=128

#GPUが使用可能かどうか確認
device = torch.device(device if torch.cuda.is_available() else "cpu")
print("device:",device)

#scycloneのGeneratorのインスタンスを生成
netG = Generator()
#学習済みモデルの読み込み
netG.load_state_dict(torch.load(scyclone_trained_model_path))
#ネットワークをデバイスに移動
netG = netG.to(device)

#Vocoderのインスタンスを生成
vocoder = Vocoder()
#学習済みモデルの読み込み
vocoder.load_state_dict(torch.load(vocoder_trained_model_path))
#ネットワークをデバイスに移動
vocoder = vocoder.to(device)

#変換対象とする音声ファイルの読み込み
input_waveform, _ = torchaudio.load(audio_path)
#読み込んだ波形からスペクトログラムを生成
input_spectrogram = torchaudio.transforms.Spectrogram(n_fft=254, hop_length=128)(input_waveform)
input_spectrogram = input_spectrogram.to(device)

#input_spectrogram : torch.Size([1, frequency, frame])
frequency = input_spectrogram.size()[-2]
frame = input_spectrogram.size()[-1]
#スペクトログラムをpaddingする際のサイズ
padding_frame = (unit_frame - cutout_frame)//2

#cutout_frameフレームずつ変換を行う　unit_frameずつinput_spectrogramから取り出しnetGで変換、出力の中央cutout_frameフレームを結果とする
result_segments = []#変換結果を格納
for i in range(0, frame//cutout_frame):
	#切り取る箇所を指定
	start_frame = i*cutout_frame - padding_frame
	end_frame = (i+1)*cutout_frame + padding_frame
	#指定の箇所を抽出
	target_segment = input_spectrogram[..., max(0, start_frame):min(frame, end_frame)]
	#足りない分に関してzero paddingを行う
	if(start_frame<0):
		target_segment = torch.cat([torch.zeros(1, frequency, -start_frame).to(device), target_segment], dim=-1)
	target_segment = torch.cat([target_segment, torch.zeros(1, frequency, unit_frame - target_segment.size()[1]).to(device)], dim=-1)
	#netGを用いて変換
	with torch.no_grad():
		result_segment = netG(target_segment)
	#出力の中央cutout_frameフレームを結果とする
	result_segments.append(torch.narrow(result_segment, dim=-1, start=16, length=cutout_frame))
#変換されたスペクトログラムを1つのTensorにまとめる
result_segments = torch.cat(result_segments, dim=-1)
#スペクトログラムのフレーム数をinput_spectrogramと同じ長さになるよう揃える
output_spectrogram = result_segments[..., 0:frame]
#スペクトログラムから負の値を取り除く
output_spectrogram = F.relu(output_spectrogram)

#Vocoderによる、スペクトログラムから波形への変換
output_waveform_by_vocoder = vocoder.generate(output_spectrogram.transpose(1, 2)).to("cpu")[None, ...]
#GriffinLimによる、スペクトログラムから波形への変換(vocoderによる生成結果との比較用)
output_waveform_by_griffinlim = torchaudio.transforms.GriffinLim(n_fft=254, n_iter=256, hop_length=128)(output_spectrogram.to("cpu"))

#音声ファイルを保存
os.makedirs(output_dir, exist_ok=True)#出力用ディレクトリがなければ作る
torchaudio.save(os.path.join(output_dir, "input_audio.wav"), input_waveform, sample_rate=16000)
torchaudio.save(os.path.join(output_dir, "output_audio_by_vocoder.wav"), output_waveform_by_vocoder, sample_rate=16000)
#比較用として、GriffinLimによって生成した波形も出力する
torchaudio.save(os.path.join(output_dir, "output_audio_by_griffinlim.wav"), output_waveform_by_griffinlim, sample_rate=16000)

#変換前後の音声を、波形とスペクトログラム2つの観点で比較するためのグラフを出力する
waveform_list = [
	(input_waveform.squeeze(dim=0).to("cpu"), "input_waveform"),
	(output_waveform_by_vocoder.squeeze(dim=0).to("cpu"), "output_waveform_by_vocoder"),
	(output_waveform_by_griffinlim.squeeze(dim=0).to("cpu"), "output_waveform_by_griffinlim")
]
spectrogram_list = [
	(input_spectrogram.squeeze(dim=0).to("cpu"), "input_spectrogram"),
	(output_spectrogram.squeeze(dim=0).to("cpu"), "output_spectrogram")
]
output_comparison_graph(
		save_path = os.path.join(output_dir, "comparison.png"),
		waveform_list=waveform_list,     #waveform_list : (torch.size([frame]), graph_title)を要素に持つlist
		spectrogram_list=spectrogram_list, #spectrogram_list : (torch.Size([frequency, frame]), graph_title)を要素に持つlist
		sampling_rate=16000 #サンプリングレート
	)

