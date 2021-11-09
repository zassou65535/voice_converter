#encoding:utf-8

import random
import numpy as np
import glob
import matplotlib as mpl
mpl.use('Agg')# AGG(Anti-Grain Geometry engine)
import matplotlib.pyplot as plt

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

#入力スペクトログラムを, scycloneのGenerator"netG"を用いて変換する
#任意のframe長のspectrogramに対応
def scyclone_inference(input_spectrogram, netG, unit_frame=160, cutout_frame=128):
	device = input_spectrogram.device
	#input_spectrogram : torch.Size([..., frequency, frame])
	frequency = input_spectrogram.size()[-2]
	frame = input_spectrogram.size()[-1]
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
	result_segments = torch.cat(result_segments, dim=-1)
	result_segments = result_segments[..., 0:frame]
	return result_segments

#音声を、波形とスペクトログラム2つの観点で比較するためのグラフを出力する
def output_comparison_graph(
		save_path, #画像の保存先
		waveform_list,     #waveform_list : (torch.size([frame]), graph_title)を要素に持つlist
		spectrogram_list, #spectrogram_list : (torch.Size([frequency, frame]), graph_title)を要素に持つlist
		sampling_rate, #サンプリングレート
	):
	plt.clf()
	plt.figure(figsize=(10,5))

	max_column_num = max(len(waveform_list), len(spectrogram_list))

	#スペクトログラムの描画
	for i in range(0, len(spectrogram_list)):
		spectrogram, graph_title = spectrogram_list[i]
		plt.subplot(2, max_column_num, i+1)
		spectrogram_db = 20*torch.log10((spectrogram+1e-10)/torch.max(spectrogram))#dbに変換, スペクトログラムを見やすくする
		plt.title(graph_title)
		plt.imshow(spectrogram_db)
		plt.xlabel("frame")
		plt.ylabel("frequency")
		plt.gca().invert_yaxis()

	#波形の描画
	for i in range(0, len(waveform_list)):
		waveform, graph_title = waveform_list[i]
		plt.subplot(2,max_column_num, max_column_num + i+1)
		num_frames = waveform.size()[0]
		time_axis = torch.arange(0, num_frames) / sampling_rate
		plt.title(graph_title)
		plt.plot(time_axis, waveform, linewidth=1)
		plt.grid()
		plt.xlabel("time")

	plt.savefig(save_path)
	plt.close()




