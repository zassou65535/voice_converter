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

#音声の波形を画像ファイルに出力
def plot_waveform(waveform, save_path, sample_rate=16000):
	#waveform : torch.size([フレーム数])
	num_frames = waveform.size()[0]
	time_axis = torch.arange(0, num_frames) / sample_rate
	plt.clf()
	plt.figure(figsize=(10,5))
	plt.plot(time_axis, waveform, linewidth=1)
	plt.grid()
	plt.xlabel("time[s]")
	plt.savefig(save_path)
	plt.clf()
	plt.close()

#音声のスペクトログラムを画像ファイルに出力
def plot_spectrogram(spectrogram, save_path, sample_rate=16000):
	#spectrogram : torch.size([周波数, フレーム数])
	#dbに変換, スペクトログラムを見やすくする
	spectrogram_db = 20*torch.log10(spectrogram/torch.max(spectrogram))
	plt.clf()
	plt.figure(figsize=(10,5))
	plt.imshow(spectrogram_db)
	plt.xlabel("frame")
	plt.ylabel("frequency")
	plt.gca().invert_yaxis()
	plt.savefig(save_path)
	plt.clf()
	plt.close()

class Audio_Dataset_for_Scyclone(data.Dataset):
	#音声のデータセットクラス
	def __init__(self, file_list, extract_frames=160, hop_length=128):
		self.file_list = file_list
		self.transform = transforms.Compose([
			torchaudio.transforms.Spectrogram(n_fft=254, hop_length=hop_length)
		])
		#音声のスペクトログラムからランダムに何フレーム切り出すか
		self.extract_frames = extract_frames
	#音声の総ファイル数を返す
	def __len__(self):
		return len(self.file_list)
	#前処理済み音声の、Tensor形式のデータを取得
	def __getitem__(self,index):
		audio_path = self.file_list[index]
		waveform, sample_rate = torchaudio.load(audio_path)
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

# #動作確認
# train_img_list = make_datapath_list("../dataset/train/domainA/jvs_extracted/ver1/jvs001/VOICEACTRESS100_010.wav")

# train_dataset = Audio_Dataset_for_Scyclone(file_list=train_img_list, extract_frames=160, hop_length=128)

# batch_size = 1
# train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

# batch_iterator = iter(train_dataloader)
# audio = next(batch_iterator)
# print(audio.size())

# waveform = torchaudio.transforms.GriffinLim(n_fft=254, n_iter=256, hop_length=128)(audio[0])[None,...]
# print(waveform.size())
# torchaudio.save("../output/test.wav", waveform, sample_rate=16000)

# plot_waveform(waveform=waveform[0], save_path="../output/waveform.png", sample_rate=16000)
# plot_spectrogram(spectrogram=audio[0], save_path="../output/spectrogram.png", sample_rate=16000)

#mu-lawアルゴリズムを適用し、波形をbit[bit]に量子化する
#参考 : https://librosa.org/doc/main/_modules/librosa/core/audio.html
def mu_raw_compression(waveform, bit):
	x = waveform
	mu = 2**bit - 1
	#mu-lawアルゴリズム
	x_compressed = torch.sign(x)*torch.log(1+mu*torch.abs(x))/np.log(1+mu)
	#波形をbit[bit]に量子化する
	x_compressed = torch.bucketize(
						input=x_compressed,
						boundaries=torch.arange(start=-1.0, end=1.0+1.0/float(mu*2), step=1.0/float(mu)), 
						right=True
					)
	return 2**(bit-1) + x_compressed
#mu_raw_compressionによって圧縮+量子化された波形を解凍する
#参考 : https://librosa.org/doc/main/_modules/librosa/core/audio.html
def mu_raw_expansion(waveform_quantized, bit):
	x = waveform_quantized - 2**(bit-1)
	mu = 2**bit - 1
	return torch.sign(x)*(1/mu)*((1+mu)**torch.abs(x)-1)

class Audio_Dataset_for_WaveRNN(data.Dataset):
	#音声のデータセットクラス
	def __init__(self, file_list, extract_frames=160, hop_length=128):
		self.file_list = file_list
		self.transform = transforms.Compose([
			torchaudio.transforms.Spectrogram(n_fft=254, hop_length=hop_length)
		])
		#音声のスペクトログラムからランダムに何フレーム切り出すか
		self.extract_frames = extract_frames
		self.hop_length = hop_length
	#音声の総ファイル数を返す
	def __len__(self):
		return len(self.file_list)
	#前処理済み音声の、Tensor形式のデータを取得
	def __getitem__(self,index):
		audio_path = self.file_list[index]
		waveform, sample_rate = torchaudio.load(audio_path)
		waveform = waveform.squeeze(dim=0)
		spectrogram = self.transform(waveform)
		#音声のスペクトログラムspectrogramからランダムにself.extract_framesフレーム切り出す
		entire_frames_length = spectrogram.size()[-1]
		start_frame = torch.randint(0, entire_frames_length-self.extract_frames, (1,))[0].item()
		end_frame = start_frame + self.extract_frames
		#spectrogramを切り取り
		spectrogram = spectrogram[..., start_frame:end_frame]
		#波形を切り取り
		waveform = waveform[..., start_frame*self.hop_length:end_frame*self.hop_length]
		print(waveform)
		#波形に対しmu-law圧縮を実行し値をbit[bit]に量子化
		waveform = mu_raw_compression(waveform=waveform, bit=10)
		print(waveform)
		print(mu_raw_expansion(waveform_quantized=waveform, bit=10))
		return waveform, spectrogram

#動作確認
train_img_list = make_datapath_list("../dataset/train/domainA/jvs_extracted/ver1/jvs001/VOICEACTRESS100_010.wav")

train_dataset = Audio_Dataset_for_WaveRNN(file_list=train_img_list, extract_frames=160, hop_length=128)

batch_size = 1
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

batch_iterator = iter(train_dataloader)
audio = next(batch_iterator)

