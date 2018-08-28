End-to-End Speech Command Recognition with Capsule Network
==========================
INTERSPEECH 2018

paper: link(to be updated)

We apply the capsule network to capture the spatial relationship and pose information of speech spectrogram features in both frequency and time axes, and show that our proposed end-to-end SR system with capsule networks on one-second speech commands dataset achieves better results on both clean and noise-added test than baseline CNN models.


Data folder structure
---
	feature_saved
	└ TEST
	  └ fbank
	    └ clean
	    └ [noise names]
	    └ [noise names]_SNR5
	  └ label
	└ TRAIN
	  └ fbank
	    └ clean
	    └ [noise names]
	  └ label
	└ VALID
	  └ fbank
	    └ clean
	    └ [noise names]
	  └ label




Authors
---
Jaesung Bae - Korea Advanced Institute of Science and Technology (KAIST)

contact: bjs2279@gmail.com / bjsd3@kaist.ac.kr
