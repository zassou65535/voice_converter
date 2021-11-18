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

def make_datapath_list(target_path):
	path_list = []#音声ファイルパスのリストを作り、戻り値とする
	for path in glob.glob(target_path, recursive=True):
		path_list.append(path)
		##読み込むパスを全部表示　必要ならコメントアウトを外す
		#print(path)
	#読み込んだ音声の数を表示
	print("audio files : " + str(len(path_list)))
	return path_list

#音声を、波形とスペクトログラム2つの観点で比較するためのグラフを出力する
def output_comparison_graph(
		save_path, #画像の保存先
		waveform_list,     #waveform_list : (torch.size([frame]), graph_title)を要素に持つlist
		spectrogram_list, #spectrogram_list : (torch.Size([frequency, frame]), graph_title)を要素に持つlist
		sampling_rate, #サンプリングレート
	):
	plt.clf()
	plt.figure(figsize=(16,5))

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

class Audio_Dataset_for_Scyclone(data.Dataset):
	#音声のデータセットクラス
	def __init__(self, file_list, augmentation, extract_frames=160, hop_length=128):
		self.file_list = file_list
		self.transform = transforms.Compose([
			torchaudio.transforms.Spectrogram(n_fft=254, hop_length=hop_length)
		])
		#音声のスペクトログラムからランダムに何フレーム切り出すか
		self.extract_frames = extract_frames
		#波形に対しデータオーギュメンテーションを適用するかどうか
		self.augmentation = augmentation
	#音声の総ファイル数を返す
	def __len__(self):
		return len(self.file_list)
	#前処理済み音声の、Tensor形式のデータを取得
	def __getitem__(self, index):
		audio_path = self.file_list[index]
		waveform, sample_rate = torchaudio.load(audio_path)

		if(self.augmentation):
			#waveformに対するdata argumentation
			#音量をランダムに1.0~2.1倍にする
			magnitude = torch.FloatTensor(1).uniform_(1.0, 2.1)
			waveform *= magnitude
			#サンプリングレートを15000~17000[Hz]の間にランダムに変更
			resampling_rate = torch.randint(15000, 17001, (1,))[0].item()
			waveform = torchaudio.transforms.Resample(16000, resampling_rate, dtype=waveform.dtype)(waveform)

		waveform = waveform.squeeze(dim=0)
		spectrogram = self.transform(waveform)
		#音声のスペクトログラムspectrogramからランダムにself.extract_framesフレーム切り出す
		entire_frames_length = spectrogram.size()[-1]
		start_frame = torch.randint(0, entire_frames_length-self.extract_frames, (1,))[0].item()
		end_frame = start_frame + self.extract_frames
		#spectrogramを切り取り
		spectrogram = spectrogram[..., start_frame:end_frame]
		#spectrogram : torch.Size([frequency, frame])
		return spectrogram

#mu-lawアルゴリズムを適用し、波形をbit[bit]に量子化する
#参考 : https://librosa.org/doc/main/_modules/librosa/core/audio.html
def mu_law_compression(waveform, bit):
	x = waveform
	mu = 2**bit-1
	#mu-lawアルゴリズム
	x_compressed = torch.sign(x)*torch.log(1+mu*torch.abs(x))/np.log(1+mu)
	#波形をbit[bit]に量子化する
	x_compressed = torch.bucketize(
						input=x_compressed,
						boundaries=torch.arange(start=-1.0, end=1.0+1.0/float(mu*4), step=2.0/float(mu)), 
						right=True
					)
	return x_compressed
#mu_law_compressionによって圧縮+量子化された波形を解凍する
#参考 : https://librosa.org/doc/main/_modules/librosa/core/audio.html
def mu_law_expansion(waveform_quantized, bit):
	mu = 2**bit-1
	x = waveform_quantized - int(mu+1)//2
	x = x*2.0/(1+mu)
	return (torch.sign(x)/mu)*(torch.pow(1+mu, torch.abs(x))-1)

class Audio_Dataset_for_Vocoder(data.Dataset):
	#音声のデータセットクラス
	def __init__(self, file_list, augmentation, extract_frames=24, hop_length=128):
		self.file_list = file_list
		self.transform = transforms.Compose([
			torchaudio.transforms.Spectrogram(n_fft=254, hop_length=hop_length)
		])
		#音声のスペクトログラムからランダムに何フレーム切り出すか
		self.extract_frames = extract_frames
		self.hop_length = hop_length
		#波形に対しデータオーギュメンテーションを適用するかどうか
		self.augmentation = augmentation
	#音声の総ファイル数を返す
	def __len__(self):
		return len(self.file_list)
	#前処理済み音声の、Tensor形式のデータを取得
	def __getitem__(self, index):
		audio_path = self.file_list[index]
		waveform, sample_rate = torchaudio.load(audio_path)

		if(self.augmentation):
			#waveformに対するdata argumentation
			#音量をランダムに1.0~2.1倍にする
			magnitude = torch.FloatTensor(1).uniform_(1.0, 2.1)
			waveform *= magnitude
			#サンプリングレートを15000~17000[Hz]の間にランダムに変更
			resampling_rate = torch.randint(15000, 17001, (1,))[0].item()
			waveform = torchaudio.transforms.Resample(16000, resampling_rate, dtype=waveform.dtype)(waveform)

		waveform = waveform.squeeze(dim=0)
		spectrogram = self.transform(waveform)
		#音声のスペクトログラムspectrogramからランダムにself.extract_framesフレーム切り出す
		entire_frames_length = spectrogram.size()[-1]
		start_frame = torch.randint(0, entire_frames_length-self.extract_frames-1, (1,))[0].item()
		end_frame = start_frame + self.extract_frames
		#spectrogramを切り取り
		spectrogram = spectrogram[..., start_frame:end_frame]
		#波形を切り取り
		waveform = waveform[..., start_frame*self.hop_length:end_frame*self.hop_length+1]
		#波形に対しmu-law圧縮を実行し値をbit[bit]に量子化
		waveform_quantized = mu_law_compression(waveform=waveform, bit=10)
		#waveform_quantized : torch.Size([frame*self.hop_length+1])
		#spectrogram : torch.Size([frequency, frame])
		return waveform_quantized, spectrogram
