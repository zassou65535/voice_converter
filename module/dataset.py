#encoding:utf-8

import random
import numpy as np
import glob

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

#音声のスペクトログラムinput_spectrogramからランダムにextract_framesフレーム切り出す
def extract_frames_randomly_from_spectrogram(input_spectrogram, extract_frames):
	entire_frames_length = input_spectrogram.size()[-1]
	start_frame = torch.randint(0, entire_frames_length-extract_frames, (1,))[0].item()
	end_frame = start_frame + extract_frames
	return input_spectrogram[..., start_frame:end_frame]

class Audio_Dataset(data.Dataset):
	#音声のデータセットクラス
	def __init__(self, file_list, extract_frames=160):
		self.file_list = file_list
		self.transform = transforms.Compose([
			torchaudio.transforms.Spectrogram(n_fft=254)
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
		waveform = self.transform(waveform)
		waveform = extract_frames_randomly_from_spectrogram(input_spectrogram=waveform, extract_frames=self.extract_frames)
		return waveform

#動作確認
# train_img_list = make_datapath_list("./sample/*.wav")

# train_dataset = Audio_Dataset(file_list=train_img_list, extract_frames=160)

# batch_size = 1
# train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

# batch_iterator = iter(train_dataloader)
# audio = next(batch_iterator)
# print(audio.size())

# waveform = torchaudio.transforms.GriffinLim(n_fft=254, n_iter=256)(audio[0])[None,...]
# print(waveform.size())
# torchaudio.save("test.wav", waveform, sample_rate=16000)




