End-to-End Speech Command Recognition with Capsule Network
==========================

INTERSPEECH 2018 paper: link(to be updated)

We apply the capsule network to capture the spatial relationship and pose information of speech spectrogram features in both frequency and time axes, and show that our proposed end-to-end SR system with capsule networks on one-second speech commands dataset achieves better results on both clean and noise-added test than baseline CNN models.

Getting Started
---
The code is implemented based on python2(2.7.12)
### Prerequistes
You should be ready to import below libraries:

	tqdm, numpy(1.14.1), termcolor, scipy, sklearn, scikits
	tensorflow(1.6.0), keras(2.1.4)


Dataset
---
We use 'Google Speech Command Dataset'. You could refer to [blog](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) and [Download Link](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)

### Adding noise
To add noise to the original dataset, we use MATLAB and [voicebox](http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html) which is MATLAB library. We run matlab code on local which is window base and upload it to server which is linux base.

1. Unzip download google speech command dataset and unzip it as name "Google_speech_command_dataset"
	    
2. Create new folder name 'Google_Speech_Command' and move command folders it. Then the folder structure will be like
```
	Google_speech_command_dataset
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

3. Run matlab code.
	

### Data folder structure
	feature_saved
	|-- TEST
	|   |-- fbank
	|   |   |-- clean
	|   |   |-- [noise names]
	|   |   '-- [noise names]_SNR5
	|   '-- label
	|-- TRAIN
	|   |-- fbank
	|   |   |-- clean
	|   |   '-- [noise names]
	|   '-- label
	'-- VALID
	    |-- fbank
	    |   |-- clean
	    |   '-- [noise names]
	    '-- label

Reference
---
Preprocessing source code from https://github.com/zzw922cn/Automatic_Speech_Recognition.

Base capsule network keras source code from https://github.com/XifengGuo/CapsNet-Keras.


Authors
---
Jaesung Bae - Korea Advanced Institute of Science and Technology (KAIST)

contact: bjs2279@gmail.com / bjsd3@kaist.ac.kr
