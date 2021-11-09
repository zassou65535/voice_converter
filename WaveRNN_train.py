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
from module.inference import *

#乱数のシードを設定　これにより再現性を確保できる
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#データセットの、各データへのパスのフォーマット　make_datapath_listへの引数
dataset_path = "./dataset/train/domainA/jvs_extracted/ver2/**/*.wav"
#学習過程を見るための、サンプル音声のパス(フォーマットではなく普通のパスとして指定)
sample_audio_path = "./dataset/train/domainA/jvs_extracted/ver2/jvs004/VOICEACTRESS100_010.wav"
#結果を出力するためのディレクトリ
output_dir = "./output/wavernn/train/"
#使用するデバイス
device = "cuda:1"
#バッチサイズ
batch_size = 16
#イテレーション数
total_iterations = 200000
#学習率
lr = 4e-4
#学習率をdecay_iterイテレーションごとにdecay_rate倍する
lr_decay_iter = 25000
lr_decay_rate = 0.5
#何イテレーションごとに学習結果を出力するか
output_iter = 2500

#出力用ディレクトリがなければ作る
os.makedirs(output_dir, exist_ok=True)

#データセットAの読み込み、データセット作成
path_list = make_datapath_list(dataset_path)
train_dataset = Audio_Dataset_for_WaveRNN(file_list=path_list, extract_frames=160)
dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=8)
print("dataset size: {}".format(len(path_list)))

#GPUが使用可能かどうか確認
device = torch.device(device if torch.cuda.is_available() else "cpu")
print("device:",device)

#ネットワークを初期化するための関数
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.xavier_uniform_(m.weight.data, gain=1.0)
		if m.bias is not None:
			m.bias.data.fill_(0.01)
	elif classname.find('Linear') != -1:
		nn.init.xavier_uniform_(m.weight.data, gain=1.0)
		if m.bias is not None:
			m.bias.data.fill_(0.01)

#Vocoderのインスタンスを生成
vocoder = WaveRNNVocoder()
#ネットワークをデバイスに移動
vocoder = vocoder.to(device)
#ネットワークの初期化
vocoder.apply(weights_init)

#optimizerをGeneratorとDiscriminatorに適用
beta1 = 0.9
beta2 = 0.999
optimizer = optim.Adam(vocoder.parameters(), lr=lr, betas=(beta1, beta2))

#Vocoderの学習過程を見るためのサンプル音声をロード、スペクトログラムを生成
sample_waveform, _ = torchaudio.load(sample_audio_path)
sample_waveform = sample_waveform.squeeze(dim=0)
sample_spectrogram = torchaudio.transforms.Spectrogram(n_fft=254, hop_length=128)(sample_waveform)
#GriffinLimによって生成したwaveform(vocoderによる生成結果との比較用)
sample_griffinlim_waveform = torchaudio.transforms.GriffinLim(n_fft=254, n_iter=256, hop_length=128)(sample_spectrogram)

#学習開始
#学習過程を追うための変数
losses = []
#現在のイテレーション回数
now_iteration = 0

print("Start Training")

#学習開始時刻を保存
t_epoch_start = time.time()

#エポックごとのループ　itertools.count()でカウンターを伴う無限ループを実装可能
for epoch in itertools.count():
	#ネットワークを学習モードにする
	vocoder.train()
	#データセットA, Bからbatch_size枚ずつ取り出し学習
	for (waveform_quantized, spectrogram) in dataloader:
		#waveform_quantized : torch.Size([frame*hop_length+1])
		#spectrogram : torch.Size([frequency, frame])
		#学習率の減衰の処理
		if((now_iteration%lr_decay_iter==0) and (not now_iteration==0)):
			optimizerG.param_groups[0]['lr'] *= lr_decay_rate
			optimizerD.param_groups[0]['lr'] *= lr_decay_rate
		#deviceに転送
		waveform_quantized = waveform_quantized.to(device)
		spectrogram = spectrogram.to(device)

		#-------------------------
 		#Vocoderの学習
		#-------------------------

		spectrogram = spectrogram.transpose(1, 2)
		predicted = vocoder(waveform_quantized[:, :-1], spectrogram)
		loss = F.cross_entropy(predicted.transpose(1, 2), waveform_quantized[:, 1:])

		#溜まった勾配をリセット
		optimizer.zero_grad()
		#傾きを計算
		loss.backward()
		#gradient explosionを避けるため勾配を制限
		nn.utils.clip_grad_norm_(vocoder.parameters(), max_norm=1.0, norm_type=2.0)
		#Generatorのパラメーターを更新
		optimizer.step()

		#グラフへの出力用
		losses.append(loss.item())

		#学習状況をstdoutに出力
		if now_iteration % 10 == 0:
			print(f"[{now_iteration}/{total_iterations}] Loss/vocoder:{loss:.5f}")

		#学習状況をファイルに出力
		if((now_iteration%output_iter==0) or (now_iteration+1>=total_iterations)):
			out_dir = os.path.join(output_dir, f"iteration{now_iteration}")
			#出力用ディレクトリがなければ作る
			os.makedirs(out_dir, exist_ok=True)

			#ここまでの学習にかかった時間を出力
			t_epoch_finish = time.time()
			total_time = t_epoch_finish - t_epoch_start
			with open(os.path.join(out_dir,"time.txt"), mode='w') as f:
				f.write("total_time: {:.4f} sec.\n".format(total_time))

			#学習済みモデル（CPU向け）を出力
			vocoder.eval()
			torch.save(vocoder.to('cpu').state_dict(), os.path.join(out_dir, "vocoder_trained_model_cpu.pth"))
			vocoder.to(device)
			vocoder.train()

			#lossのグラフ(対数スケール)を出力
			plt.clf()
			plt.figure(figsize=(10, 5))
			plt.title("Vocoder Loss During Training")
			plt.plot(losses, label="loss")
			plt.xlabel("iterations")
			plt.ylabel("Loss")
			plt.legend()
			plt.grid()
			plt.savefig(os.path.join(out_dir, "loss.png"))
			plt.close()

			#推論を実行、結果を保存する
			#推論を実行
			vocoder.eval()
			sample_generated_waveform = vocoder.generate(sample_spectrogram[None, ...].transpose(1, 2).to(device))
			vocoder.train()
			#結果を保存する
			torchaudio.save(os.path.join(out_dir, "sample_audio.wav"), sample_waveform[None, ...], sample_rate=16000)
			torchaudio.save(os.path.join(out_dir, "sample_generated_audio.wav"), sample_generated_waveform[None, ...], sample_rate=16000)
			#比較用として、GriffinLimによって生成したwavも出力する
			torchaudio.save(os.path.join(out_dir, "sample_griffinlim_audio.wav"), sample_griffinlim_waveform[None, ...], sample_rate=16000)

			#音声を、波形とスペクトログラム2つの観点で比較するためのグラフを出力する
			waveform_list = [
				(sample_waveform, "original_waveform"),
				(sample_generated_waveform, "waveform generated by waveRNN"),
				(sample_griffinlim_waveform, "waveform generated by GriffinLim"),
			]
			spectrogram_list = [
				(sample_spectrogram, "spectrogram"),
			]
			output_comparison_graph(
					save_path = os.path.join(out_dir, "comparison.png"),
					waveform_list=waveform_list,     #waveform_list : (torch.size([frame]), graph_title)を要素に持つlist
					spectrogram_list=spectrogram_list, #spectrogram_list : (torch.Size([frequency, frame]), graph_title)を要素に持つlist
					sampling_rate=16000, #サンプリングレート
				)

		now_iteration += 1
		#イテレーション数が上限に達したらループを抜ける
		if(now_iteration>=total_iterations):
			break
	#イテレーション数が上限に達したらループを抜ける
	if(now_iteration>=total_iterations):
		break
