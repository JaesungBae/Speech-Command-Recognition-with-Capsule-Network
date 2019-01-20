End-to-End Speech Command Recognition with Capsule Network
==========================

INTERSPEECH 2018 paper: [link](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1888.pdf)

We apply the capsule network to capture the spatial relationship and pose information of speech spectrogram features in both frequency and time axes, and show that our proposed end-to-end SR system with capsule networks on one-second speech commands dataset achieves better results on both clean and noise-added test than baseline CNN models.

* **20 JAN 2019:** Other baseline Keyword Spotting(KWS) models are also provided in CNN code.

Getting Started
---
The code is implemented based on python2(2.7.12)
### Prerequistes
You should be ready to import below libraries:

	tqdm, numpy(1.14.1), termcolor, scipy, sklearn, scikits
	tensorflow(1.6.0), keras(2.1.4)

	pip install numpy
	pip install termcolor
	pip install scipy
	pip install sklearn
	pip install scikit-learn
	pip install tensorflow-gpu==1.6.0
	pip install keras==2.1.4
	
Speech Feature Generation
---
### Dataset
We use 'Google Speech Command Dataset'. You could refer to [blog](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) and [Download Link](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)

- Download the dataset from above link and unzip it. (In our case we will unzip it in the folder named 'Google_Speech_Command')

### Adding noise
To add noise to the original dataset, we use MATLAB and [voicebox](http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html) which is MATLAB library. We run matlab code on local which is window base and upload it to server which is linux base.

1. Unzip download google speech command dataset.
	    
2. Create new folder name 'Google_Speech_Command' and move command folders into it. Then the folder structure will be like
```
speech_commands_v0.01.tar
|-- [_backgorund_noise_]
|-- Google_Speech_Command
|   |-- bed
|   |-- bird
 :      :
|   '-- zero
|-- testing_list
|-- validation_list
'-- etc.
```

3. Change 'data_path' in matlab code and run the matlab code. It will generate new folder and save the noise added audio files.
```	
noise_wave_generate.m
```
4. You could aslo change 'SNR' in the code and generate noise audio files as you want.

### Feature Generation
Extract speech features from raw audio file and save them as .npy file. Please adjust '--noise_name' argument.
```
cd core
python feature_generation.py
```

#### Data folder structure
	feature_saved
	|-- TEST
	|   |-- fbank
	|   |   |-- clean
	|   |   '-- [noise names]_SNR5
	|   '-- label
	|-- TRAIN
	|   |-- fbank
	|   |   |-- clean
	|   |   '-- [noise names]_SNR5
	|   '-- label
	'-- VALID
	    |-- fbank
	    |   |-- clean
	    |   '-- [noise names]_SNR5
	    '-- label

Training & Testing
---
For training and testing go into 'CNN' or 'CapsNet' folder and run the code. You could change the mode with '--is_training' argument.
### Training
```
cd CapsNet
python main.py -m=CapsNet -ex='0320_digitvec4' -d=0 --kernel=19 --primary_channel=32  --primary_veclen=4 --digit_veclen=4 --is_training='TRAIN'  --batch_size=64
```

### Testing
Note that you should set '--keep' argument to the number of epoch that you want to test.
```
cd CapsNet
python main.py -m=CapsNet -ex='0320_digitvec4' -d=0 --kernel=19 --primary_channel=32  --primary_veclen=4 --digit_veclen=4 --is_training='TEST' --batch_size=64 -tr='clean' -te='clean' --SNR=5 --keep=?
```

### Various Neural Networks base KWS models
KWS models based on various kinds of Neural Networks(NNs) are also provided in CNN/model.py

1. Deep Neural Network(DNN) base KWS model from 
  - G. Chen, C. Parada, and G. Heigold, “Small-footprint keyword spotting using deep neural networks.” in *ICASSP*, vol. 14. Citeseer, 2014, pp. 4087–4091.

2. CNN base KWS model from
  - T. N. Sainath and C. Parada, “Convolutional neural networks for small-footprint keyword spotting,” in *Sixteenth Annual Conference of the International Speech Communication Association*, 2015.

3. Long Short-Term Memory(LSTM) base KWS model form
  - M. Sun, A. Raju, G. Tucker, S. Panchapagesan, G. Fu, A. Mandal, S. Matsoukas, N. Strom, and S. Vitaladevuni, “Max-pooling loss training of long short-term memory networks for small-footprint keyword spotting,” in *Spoken Language Technology Workshop (SLT)*, 2016 IEEE. IEEE, 2016, pp. 474–480.

4. Convolutional Recurrent Neural Network(CRNN) base KWS model from
  - S. O. Arik, M. Kliegl, R. Child, J. Hestness, A. Gibiansky, C. Fougner, R. Prenger, and A. Coates, “Convolutional recurrent neural networks for small-footprint keyword spotting,” *arXiv preprint arXiv:1703.05390*, 2017.

Reference
---
Preprocessing source code from https://github.com/zzw922cn/Automatic_Speech_Recognition.

Base capsule network keras source code from https://github.com/XifengGuo/CapsNet-Keras.


Authors
---
Jaesung Bae - Korea Advanced Institute of Science and Technology (KAIST)

contact: bjs2279@gmail.com
