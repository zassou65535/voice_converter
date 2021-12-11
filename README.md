# voice_converter

## 概要
Pytorchによる、<a href="https://arxiv.org/abs/2005.03334">Scyclone</a>と<a href="https://github.com/anandaswarup/waveRNN">Vocoder</a>の2つを用いた音声変換器です。  
詳しい解説と音声変換例は<a href="">こちら</a>。

## 想定環境
Ubuntu20.04  
python 3.8.5 + torch==1.9.1+cu111 + torchaudio==0.9.1  
ライブラリの詳細は`requirements.txt`を参照。

## プログラム
VocoderとScycloneの学習はそれぞれ別々に行います。
### Vocoder
* `Vocoder_train.py`はVocoderの学習を実行、学習の過程と学習済みモデルを出力するプログラムです。  
* `Vocoder_inference.py`は`Vocoder_train.py`によって出力された学習済みVocoderを読み込み、推論(スペクトログラムから音声波形の生成)を実行、結果を出力するプログラムです。
### Scyclone
* `Scyclone_train.py`はScycloneの学習を実行、学習の過程と学習済みモデルを出力するプログラムです。
* `Scyclone_inference.py`は`Vocoder_train.py`によって出力された学習済みVocoderと、`Scyclone_train.py`によって出力された学習済みGeneratorの2つを読み込み、`.wav`ファイルに対し推論(ドメインA(変換元)からドメインB(変換先)への変換)を実行し結果を出力するプログラムです。

## データセットに関する注意点
データセットはサンプリングレート16000[Hz]、長さ約1.5秒以上の`.wav`形式のファイル群を想定しています。  
また、データセットサイズはドメインA(変換元)、ドメインB(変換先)それぞれで少なくとも5000以上とすることを強く推奨します。

## 使い方(Vocoder)
### 学習の実行
1. `Vocoder_train.py`の32行目付近の変数`dataset_path`で音声ファイル群のパスの形式を指定します。
1. `Vocoder_train.py`の35行目付近の変数`sample_audio_path`で、学習過程を見るための、サンプルとなる音声ファイルのパスを指定します。
1. `Vocoder_train.py`の置いてあるディレクトリで`python Vocoder_train.py`を実行することで学習を実行します。
	* 学習の過程が`./output/vocoder/train/`以下に出力されます。
	* 学習済みVocoderが`./output/vocoder/train/iteration150000/vocoder_trained_model_cpu.pth`などという形で5000イテレーション毎に出力されます。
### 推論の実行
1. `Vocoder_inference.py`の33行目付近の変数`audio_path`で対象とする`.wav`ファイルのパスを指定します。
1. `Vocoder_inference.py`の35行目付近の変数`vocoder_trained_model_path`で学習済みVocoderへのパス(例えば`./output/vocoder/train/iteration150000/vocoder_trained_model_cpu.pth`など)を指定します。
1. `Vocoder_inference.py`の置いてあるディレクトリで`python Vocoder_inference.py`を実行して、`audio_path`で指定した`.wav`ファイルに対し推論を行います。
	* 「音声波形(`.wav`)→スペクトログラム→音声波形」と実行され、結果が`./output/vocoder/inference/`以下に出力されます。

## 使い方(Scyclone)
以下ではドメインA(変換元)、ドメインB(変換先)をそれぞれ単にA、Bと呼称します。
### 学習の実行
1. `Scyclone_train.py`の34行目付近の変数`dataset_path_A`でAに属する、`dataset_path_B`でBに属する音声ファイル群のパスの形式を指定します。
1. `Scyclone_train.py`の置いてあるディレクトリで`python Scyclone_train.py`を実行することで「A⇄B」の変換ができるよう学習を実行します。
	* 学習の過程が`./output/scyclone/train/`以下に出力されます。
	* 学習済みGeneratorが`./output/scyclone/train/iteration380000/generator_A2B_trained_model_cpu.pth`などという形で5000イテレーション毎に出力されます。
### 推論の実行
1. `Scyclone_inference.py`の34行目付近の変数`audio_path`で変換対象とする`.wav`ファイルのパスを指定します。
1. `Scyclone_inference.py`の36行目付近の変数`scyclone_trained_model_path`で学習済みGeneratorへのパスを指定します。
1. `Scyclone_inference.py`の38行目付近の変数`vocoder_trained_model_path`で学習済みVocoderへのパスを指定します。
1. `Scyclone_inference.py`の置いてあるディレクトリで`python Scyclone_inference.py`を実行して、`audio_path`で指定した`.wav`ファイルに対し推論を行います。
	* `.wav`ファイルに対し「A→B」と実行され、結果が`./output/scyclone/inference/`以下に出力されます。

## 参考
<a href="https://github.com/tarepan/Scyclone-PyTorch">Reimplmentation of voice conversion system "Scyclone" with PyTorch</a>  
<a href="https://github.com/anandaswarup/waveRNN">Recurrent Neural Network based Neural Vocoders</a>