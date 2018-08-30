import os, sys
sys.path.append('../')
import numpy as np
from termcolor import cprint
from random import randint
from tqdm import tqdm
# Audio library
from sklearn import preprocessing
import scipy.io.wavfile as wav
from scikits.audiolab import Format, Sndfile
from scikits.audiolab import wavread
import math
# User(ref: )
from core.calcmfcc import calcfeat_delta_delta

# "Bed", "Bird", "Cat", "Dog", "Happy", "House", "Marvin", "Sheila", "Tree", and "Wow"
# 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']

def text_to_label(text):
    return {'bed':0, 'bird':1, 'cat':2, 'dog':3, 'down':4, 
        'eight':5, 'five':6, 'four':7, 'go':8, 'happy':9, 
        'house':10, 'left':11, 'marvin':12, 'nine':13, 'no':14, 
        'off':15, 'on':16, 'one':17, 'right':18, 'seven':19, 
        'sheila':20, 'six':21, 'stop':22, 'three':23, 'tree':24,
        'two':25, 'up':26, 'wow':27, 'yes':28, 'zero':29}.get(text, 30)
    #return {'down': 2,'go':3,'left':4,'no':5,'off':6,'on':7,'right':8,'stop':9,'up':10,'yes':11}.get(text, 1) #1 is unknown

class CSV_saver(object):
    def __init__(self,ex_name,csv_reset=True):
        self.ex_name = ex_name
        self.csv_reset = csv_reset

        self.result_path,self.save_path = self.path()
        self.fd_loss, self.fd_val_result, self.fd_val_acc = self.save_to()

    def path(self):
        # make save and result path
        result_path = os.path.join('./results', self.ex_name)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        save_path = os.path.join('./save', self.ex_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        return result_path, save_path

    def get_path(self):
        # get save and result path.
        return self.result_path, self.save_path

    def save_to(self):
        # make csv file and open it.
        loss = self.result_path + '/loss.csv'
        val_result = self.result_path + '/val_result.csv'
        val_acc = self.result_path + '/val_acc.csv'
        if self.csv_reset:
            fd_val_result = open(val_result, 'a')
            fd_loss = open(loss, 'a')
            fd_val_acc = open(val_acc, 'a') # not overwrite
            fd_val_result.write('continue learning\n')
            fd_loss.write('continue learning\n')
            fd_val_acc.write('continue learning\n')
        else:
            print('**** Remove the csv files ****')
            if os.path.exists(val_acc):
                os.remove(val_acc)
            if os.path.exists(loss):
                os.remove(loss)
            if os.path.exists(val_result):
                os.remove(val_result)
            fd_loss = open(loss,'w')
            fd_val_result = open(val_result,'w')
            fd_val_acc = open(val_acc,'w')
            fd_val_result.write('step,val_result\n')
            fd_loss.write('step,loss\n')
            fd_val_acc.write('step,val_acc\n')
        return fd_loss, fd_val_result, fd_val_acc
    
    def write(self,step,data,file):
        # write values in csv file.
        if file == 'loss':
            self.fd_loss.write(str(step) + ',' + str(data) + "\n")
            self.fd_loss.flush()
        elif file == 'val_result':
            # wirte real and decoded sentence.
            decoded_sentence, real_sentence = data
            # decoded_sentence and real_sentence should be str.
            self.fd_val_result.write(str(step)+"\n"+ decoded_sentence + '\n' + real_sentence + "\n")
            self.fd_val_result.flush()
        elif file == 'val_acc':
            self.fd_val_acc.write(str(step) + ',' + str(data) + "\n")
            self.fd_val_acc.flush()
        else:
            raise ValueError

    def close(self):
        # close the csv file at the end of code.
        self.fd_loss.close()
        self.fd_val_result.close()
        self.fd_val_acc.close()

def SNR(sig,noise=None):
    power = np.power(np.array(sig),2)
    power = np.mean(power)
    #print power
    sigdB = 20 * np.log10(power)
    #print(np.log10(power[1]),sigdB[1])
    #print sigdB
    noise_power = np.power(np.array(noise),2)
    noise_power = np.mean(noise_power)
    noisedB = 20 * np.log10(noise_power)
    #print(sigdB,noisedB)
    SNR = sigdB - noisedB
    return SNR

def feature_generation(audio_path, save_path, win_length=0.02, win_step=0.01, mode='fbank', 
		feature_len=40, noise_name='clean', noiseSNR=0.5,csv=None):
    '''
    <input>
    audio_path = '/sda3/DATA/jsbae/Google_Speech_Command'
    save_path = '/home/jsbae/STT2/KWS/feature_saved'
    win_length: default=0.02, "specify the window length of feature"
    win_step: default=0.01, "specify the window step length of feature"
    mode: choices=['mfcc', 'fbank']
    feature_len: default=40,'Features length'
    <output>
    No output. Save featuere and label(int) to npy filetpye.
    '''
    # Read testing_list and validation_list and make it to likst.
    testing_list_path = os.path.join(audio_path, 'testing_list.txt')
    validation_list_path = os.path.join(audio_path, 'validation_list.txt')
    def read_txt(path):
        text_file = open(path,'r')
        lines = text_file.read().split('\n')
        return lines
    testing_list = read_txt(testing_list_path)
    validation_list = read_txt(validation_list_path)
    # end.
    #
    # make dirs.
    if noise_name == 'clean':
        dirs = [f for f in os.listdir(audio_path) if os.path.isdir(os.path.join(audio_path, f))] #label save at dirs
        dirs.sort()
        if '_background_noise_' in dirs:
            dirs.remove('_background_noise_')
    else:
        audio_path = os.path.join(audio_path,noise_name)
        dirs = [f for f in os.listdir(audio_path) if os.path.isdir(os.path.join(audio_path, f))] #label save at dirs
        '''
        dirs=[]
        dirs1 = [f for f in os.listdir(audio_path) if os.path.isdir(os.path.join(audio_path, f))] #label save at dirs
        dirs1.sort()
        for dirs1_ in dirs1:
            dirs1_path = os.path.join(audio_path,dirs1_)
            dirs2 = [f for f in os.listdir(dirs1_path) if os.path.isdir(os.path.join(dirs1_path,f))]
            for dirs2_ in dirs2:
                dirs.append(dirs1_ + '/' + dirs2_)
        '''
    print(dirs)
    print('Number of labels: ' + str(len(dirs)))
    # end.
    #
    # Make directory if not exits.
    noise_name = noise_name + '_SNR' + str(noiseSNR)# change noise_name with SNR
    def make_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
    make_dir(os.path.join(save_path,'TEST',mode, noise_name))
    make_dir(os.path.join(save_path,'VALID',mode, noise_name))
    make_dir(os.path.join(save_path,'TRAIN',mode, noise_name))
    make_dir(os.path.join(save_path,'TEST','label'))
    make_dir(os.path.join(save_path,'VALID','label'))
    make_dir(os.path.join(save_path,'TRAIN','label'))
    #end.
    #
    ############# TO COMPUTE TOTAL SNR #############
    total_SNR = 0
    count = 0
    # Make feature and label files.
    for dirname in dirs:
        full_dirname = os.path.join(audio_path,dirname)
        cprint('Processing in '+full_dirname,'yellow')
        teCount, vaCount, trCount = 0,0,0
        for filename in os.listdir(full_dirname):
            full_filename = os.path.join(full_dirname,filename)
            #print full_filename
            filenameNoSuffix =  os.path.splitext(full_filename)[0]
#            print filenameNoSuffix: output: /sda3/DATA/jsbae/Google_Speech_Command/nine/24b82192_nohash_2
            ext = os.path.splitext(full_filename)[-1]
            if ext == '.wav':
                #print dirname #label
                #print full_filename #wavfile path
                rate = None
                sig = None
                
                
                #(rate,sig)= wav.read(full_filename)
                '''
                else:
                    (c_rate,c_sig)= wav.read(full_filename)
                    (n_rate,n_sig)= wav.read(bgmfile)
                    assert c_rate == n_rate ==  16000
                    rate = c_rate
                    n_startpoint = randint(0,len(n_sig)-len(c_sig))
                    while n_startpoint == 0:
                        cprint('n_startpoint 0','red')
                        n_startpoint = randint(0,len(n_sig)-len(c_sig))
                    sig = (1-noiseSNR) * c_sig + noiseSNR * n_sig[n_startpoint:n_startpoint+len(c_sig)]
                    
                    snr_ = SNR(sig,n_sig[n_startpoint:n_startpoint+len(c_sig)]*noiseSNR)
                    total_SNR += snr_
                    count += 1
                    csv.write(str(count)+','+str(snr_)+'\n')
                    csv.flush()
                '''

                
                try:
                    (rate,sig)= wav.read(full_filename)
                except ValueError as e:
                    if e.message == "File format 'NIST'... not understood.":
                        sf = Sndfile(full_filename, 'r')
                        nframes = sf.nframes
                        sig = sf.read_frames(nframes)
                        rate = sf.samplerate
                
                feat = calcfeat_delta_delta(sig,rate,win_length=win_length,win_step=win_step,mode=mode,feature_len=feature_len)
                feat = preprocessing.scale(feat)
                feat = np.transpose(feat) 
                #print(np.max(feat),np.min(feat),feat.shape) #(120, almost 99)
                label = text_to_label(dirname)
                #print label
                if label == 30: raise ValueError('wrong') 
                # Save to TEST, VALID, TRAIN folder.
                if os.path.join(dirname,filename) in testing_list:
                    featureFilename = os.path.join(save_path,'TEST',mode, noise_name, dirname+'_'+filenameNoSuffix.split('/')[-1]+'.npy')
                    labelFilename = os.path.join(save_path,'TEST','label', dirname+'_'+filenameNoSuffix.split('/')[-1]+'.npy')
                    assert label == np.load(labelFilename)
                    print featureFilename
                    np.save(featureFilename, feat)
                    #np.save(labelFilename, label)
                    teCount +=1

                elif os.path.join(dirname,filename) in validation_list:
                    featureFilename = os.path.join(save_path,'VALID',mode, noise_name, dirname+'_'+filenameNoSuffix.split('/')[-1]+'.npy')
                    labelFilename = os.path.join(save_path,'VALID','label', dirname+'_'+filenameNoSuffix.split('/')[-1]+'.npy')
                    #np.save(featureFilename, feat)
                    #np.save(labelFilename, label)
                    vaCount += 1
                    raise ValueError
                else:
                    featureFilename = os.path.join(save_path,'TRAIN',mode, noise_name, dirname+'_'+filenameNoSuffix.split('/')[-1]+'.npy')
                    labelFilename = os.path.join(save_path,'TRAIN','label', dirname+'_'+filenameNoSuffix.split('/')[-1]+'.npy')
                    #np.save(featureFilename, feat)
                    #np.save(labelFilename, label)
                    trCount +=1
                    raise ValueError
        print trCount, vaCount, teCount
    # end.
    # function end.

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocessing for Google Speech Command dataset.")
    parser.add_argument('--win_length', default=0.02, type=float)
    parser.add_argument('--win_step', default=0.01, type=float)
    parser.add_argument('--mode', default='fbank',choices=['mfcc','fbank'], type=str)
    parser.add_argument('--feature_len', default=40, type=int)
    parser.add_argument('--noise_name', default='clean', 
        choices=['clean','exercise_bike','pink_noise','doing_the_dishes','running_tap','dude_miaowing','white_noise'],
        type=str)
    parser.add_argument('--noiseSNR', default=5, type=int)

    # Parameters
    args = parser.parse_args()
    win_length = args.win_length
    win_step = args.win_step
    mode = args.mode
    feature_len = args.feature_len
    noise_name = args.noise_name
    noiseSNR = args.noiseSNR

    audio_path = '/DATA/jsbae/Google_Speech_Command'
    save_path = '/DATA/jsbae/KWS_feature_saved'

    ############### FOR FEATURE EXTRACTION ###############
    # Path and function
    '''
    cprint('noise name: '+noise_name,'magenta')
    cprint('save_path: '+save_path+'\nmode: '+mode,'yellow')
    feature_generation(audio_path=audio_path, save_path=save_path,    
        win_length=win_length, win_step=win_step,mode=mode,feature_len=feature_len,
        noise_name=noise_name, noiseSNR=noiseSNR)
	'''

    ########### For Noise SNR test ###########
    audio_path = audio_path + '_SNR' + str(args.noiseSNR)
    noise = ['exercise_bike','pink_noise','doing_the_dishes','running_tap','dude_miaowing','white_noise']
    snr_memory = []
    for i in range(len(noise)):
    	snr_csv = './'+str(noise[i])+'.csv'
    	fd_snr = open(snr_csv ,'w')
    	fd_snr.write('count,snr\n')
    	print(noise[i])
        total_snr = feature_generation(audio_path=audio_path, save_path=save_path,    
            win_length=win_length, win_step=win_step,mode=mode,feature_len=feature_len,
            noise_name=noise[i], noiseSNR=noiseSNR,csv=fd_snr)
        snr_memory.append(total_snr)
        fd_snr.close()
    for i in range(len(noise)):
        print('The SNR for the noise "{}" is: {}'.format(noise[i],snr_memory[i]))