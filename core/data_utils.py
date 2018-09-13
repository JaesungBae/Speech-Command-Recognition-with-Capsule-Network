import os, sys
sys.path.append('../')
from random import randint
import numpy as np
from termcolor import cprint
from keras.utils import to_categorical

'''
if mode is fbank we assume that featLength is 40.
and if mode is mfcc we assume that featLength is 13.
'''
def load_random_noisy_data(data_path, is_training, mode, SNR=None):
    if is_training != 'TEST' and SNR != None:
        raise ValueError('There is no SNR noise generated for the TRAIN and VALID case.')

    if SNR is not None:
        featPath = os.path.join(data_path,is_training,mode) #/TRAIN/fbank
        labelPath = os.path.join(data_path,is_training,'label')#/TRAIN/label
    else:
        featPath = os.path.join(data_path,is_training,mode) #/TRAIN/fbank
        labelPath = os.path.join(data_path,is_training,'label')#/TRAIN/label
    
    def Generate_noise_list(featPath, SNR):
        '''
        Generate noise list. EX: ['doing_the_dishes_SNR5']
        '''
        noise_list = os.listdir(featPath)
        print noise_list
        out = []
        if SNR is not None:
            for ii in noise_list:
                #print ii.split('_')[-1], ii.split('_')[-1]=='SNR'+str(SNR)
                if ii.split('_')[-1] == 'SNR'+str(SNR):
                    print ii
                    out.append(ii)
            print out
            return out
        else:                
            noise_list.remove('clean')
            return noise_list

    noise_list = Generate_noise_list(featPath,SNR)
    assert len(noise_list) == 6
    if mode == 'fbank':
        featLength = 40
    elif mode == 'mfcc':
        featLength = 13
    else:
        raise ValueError('wrong mode name')

    sampleLength = len(os.listdir(os.path.join(featPath,'clean')))
    x = np.zeros((sampleLength,99,featLength*3))
    y = np.zeros((sampleLength,))
    for index, filename in enumerate(os.listdir(os.path.join(featPath,'clean'))):
        #print os.path.join(featPath, noise_list[randint(0,5)], filename)
        FeatArray = np.load(os.path.join(featPath, noise_list[randint(0,5)], filename)).T
        if FeatArray.shape[0] < 99:
            padSec = 99 - FeatArray.shape[0]
            FeatArray = np.pad(FeatArray, ((0,padSec),(0,0)), 'constant', constant_values=0)
        elif FeatArray.shape[0] >99:
            raise ValueError('Maxtime step wrong!')
        x[index, :,:] = FeatArray
        y[index] = np.load(os.path.join(labelPath, filename))
    # [sample,99,120]
    x_new = np.zeros((x.shape[0],99,featLength,3))
    x_new[:,:,:,0] = x[:,:,:featLength]
    x_new[:,:,:,1] = x[:,:,featLength:2*featLength]
    x_new[:,:,:,2] = x[:,:,2*featLength:]
    y = to_categorical(y.astype('float32'))
    #y = np.expand_dims(y,axis=1)
    return x_new,y

def load_specific_noisy_data(saved_path, is_training, mode, noise_name):
    mfccPath = os.path.join(saved_path,is_training,mode,noise_name)
    labelPath = os.path.join(saved_path,is_training,'label')

    if mode == 'fbank':
        featLength = 40
    elif mode == 'mfcc':
        featLength = 13
    else:
        raise ValueError('wrong mode name')

    sampleLength = len(os.listdir(mfccPath))
    x = np.zeros((sampleLength,99,3*featLength))
    y = np.zeros((sampleLength,))
    print('start')
    for index, filename in enumerate(os.listdir(mfccPath)):
        FeatArray = np.load(os.path.join(mfccPath, filename)).T
        if FeatArray.shape[0] < 99:
            padSec = 99 - FeatArray.shape[0]
            FeatArray = np.pad(FeatArray, ((0,padSec),(0,0)), 'constant', constant_values=0)
        elif FeatArray.shape[0] >99:
            raise ValueError('Maxtime step wrong!')
        x[index, :,:] = FeatArray
        y[index] = np.load(os.path.join(labelPath, filename))
    print('end')
    # [sample,99,120]
    x_new = np.zeros((x.shape[0],99,featLength,3))
    x_new[:,:,:,0] = x[:,:,:featLength]
    x_new[:,:,:,1] = x[:,:,featLength:2*featLength]
    x_new[:,:,:,2] = x[:,:,2*featLength:]
    y = to_categorical(y.astype('float32'))
    #y = np.expand_dims(y,axis=1)
    return x_new,y

def Dimension(data,dimension):
    '''
    data shape: [Number,99(time),40(feat),3]
    dimension: if dimension is smaller than 3 -> choose one dimension from data.
                if dimension is 3 -> use full dimension.
    '''
    if dimension < 3:
        data = np.expand_dims(data[:,:,:,1],axis=3)
    elif dimension == 3:
        pass
    else:
        raise ValueError('Dimension value wrong')
    return data

def DATA(is_training, train_with, test_with, data_path, mode, dimension):
    '''
    The first trX will have size [Number,99(time),40(feat),3]
    make it to channel 1 or 3
    '''
    if is_training == 'TRAIN':
        if train_with == 'clean':
            trX, trY = load_specific_noisy_data(data_path,is_training='TRAIN',mode=mode,noise_name='clean')
        elif train_with == 'noisy':
            raise ValueError('To be updated')
        else:
            raise ValueError('should set "--train_with" value')
        #
        if test_with == 'clean':
            vaX, vaY = load_specific_noisy_data(data_path,is_training='VALID',mode=mode,noise_name='clean')
        elif test_with == 'noisy':
            vaX, vaY = load_random_noisy_data(data_path,'VALID',mode)
        else:
            raise ValueError('should set "--test_with" value')
        #
        trX = Dimension(trX,dimension)
        vaX = Dimension(vaX,dimension)
        print(str(trX.shape),str(trY.shape),str(vaX.shape),str(vaY.shape))
        return (trX, trY, vaX, vaY)
    elif is_training =='TEST':
        teX, teY = load_specific_noisy_data(data_path,is_training='TEST',mode=mode,noise_name='clean') 
        # dimension start
        teX = Dimension(teX,dimension)
        print(str(teX.shape),str(teY.shape))
        return (teX, teY) #shape [sample, 120(f), 99?(t)], [sample,]

def DATA_capsule(is_training, train_with, test_with, data_path, mode):
    '''
    The first trX will have size [Number,99(time),40(feat),3]
    make it to channel 1 or 3
    '''
    if is_training == 'TRAIN':
        if train_with == 'clean':
            trX, trY = load_specific_noisy_data(data_path,is_training='TRAIN',mode=mode,noise_name='clean')
        elif train_with == 'noisy':
            trX1, trY1 = load_specific_noisy_data(data_path,is_training='TRAIN',mode=mode,noise_name='clean')
            print(str(trX1.shape),str(trY1.shape))
            trX2, trY2 = load_specific_noisy_data(data_path,is_training='TRAIN',mode=mode,noise_name='white_noise')
            print(str(trX2.shape),str(trY2.shape))
            trX3, trY3 = load_specific_noisy_data(data_path,is_training='TRAIN',mode=mode,noise_name='pink_noise')
            print(str(trX3.shape),str(trY3.shape))
            trX = np.concatenate((trX1,trX2,trX3),axis=0)
            trY = np.concatenate((trY1,trY2,trY3),axis=0)
        else:
            raise ValueError('should set "--train_with" value')
        if test_with == 'clean':
            vaX, vaY = load_specific_noisy_data(data_path,is_training='VALID',mode=mode,noise_name='clean')
        elif test_with == 'noisy':
            vaX, vaY = load_random_noisy_data(data_path,'VALID',mode)
        else:
            raise ValueError('should set "--test_with" value')
        #trX = np.expand_dims(trX[:,:,:,1],axis=3)
        #vaX = np.expand_dims(vaX[:,:,:,1],axis=3)
        print(str(trX.shape),str(trY.shape),str(vaX.shape),str(vaY.shape))
        return (trX, trY, vaX, vaY)
    elif is_training =='TEST':
        teX, teY = load_specific_noisy_data(data_path,is_training='TEST',mode=mode,noise_name='clean') 
        #teX = np.expand_dims(teX[:,:,:,1],axis=3)
        print(str(teX.shape),str(teY.shape))
        return (teX, teY) #shape [sample, 120(f), 99?(t)], [sample,]

def DATA_CNN(is_training, train_with, test_with, data_path, mode):
    if is_training == 'TRAIN':
        if train_with == 'clean':
            trX, trY = load_specific_noisy_data(data_path,is_training='TRAIN',mode=mode,noise_name='clean')
        elif train_with == 'noisy':
            trX, trY = load_random_noisy_data(data_path,'TRAIN',mode)
        else:
            raise ValueError('should set "--train_with" value')
        if test_with == 'clean':
            vaX, vaY = load_specific_noisy_data(data_path,is_training='VALID',mode=mode,noise_name='clean')
        elif test_with == 'noisy':
            vaX, vaY = load_random_noisy_data(data_path,'VALID',mode)
        else:
            raise ValueError('should set "--test_with" value')
        #trX = np.expand_dims(trX[:,:,:,1],axis=3)
        #vaX = np.expand_dims(vaX[:,:,:,1],axis=3)
        print(str(trX.shape),str(trY.shape),str(vaX.shape),str(vaY.shape))
        return (trX, trY, vaX, vaY)
    elif is_training =='TEST':
        teX, teY = load_specific_noisy_data(data_path,is_training='TEST',mode=mode,noise_name='clean') 
        #teX = np.expand_dims(teX[:,:,:,1],axis=3)
        print(str(teX.shape),str(teY.shape))
        return (teX, teY) #shape [sample, 120(f), 99?(t)], [sample,]




























    #cprint('padding: ' + str(args.padding), 'cyan')
'''
def mean_normalization(data):
        data -= (np.mean(data, axis=0) + 1e-8)
        return data
'''

'''
saved_path = '/home/jsbae/STT2/KWS/feature_saved'
#x,y=load_random_noisy_data(saved_path,'TRAIN','fbank')
x,y=load_specific_noisy_data(saved_path,'TRAIN','fbank',noise_name='clean')
print x.shape
print y.shape
'''


"""
def load_specific_noise_data(saved_path, is_training, mode, noise_name):
    

    def preprocess(inputList, targetList):
        assert len(inputList) == len(targetList)
        sampleLength = len(inputList)
        nFeatures = inputList[0].shape[0]
        def max_length(inputList):
            maxLength = 0
            for inp in inputList:
                maxLength = max(maxLength, inp.shape[-1])
            return maxLength
        x_maxLength = max_length(inputList)
        cprint('x_maxLength: ',str(x_maxLength),'yellow')
        x = np.zeros((sampleLength, x_maxLength, nFeatures))
        y = np.zeros((sampleLength, ))


        for index, val in enumerate(inputList):
            x[index,:,:] = inputList[index].T

        for index, val in enumerate(targetList):
            y[index] = targetList[index]
        return x,y
        '''
        padSec = x_maxLength - inputList[index].shape[1]
        x_train[index, :, :] = np.pad(inputList[index].T, ((0,padSec),(0,0)), 'constant', constant_values=0)
        x_seqlen[index,:] = inputList[index].shape[-1]
        '''
    mfccPath = os.path.join(saved_path,is_training,mode,noise_name)
    labelPath = os.path.join(saved_path,is_training,'label')

    # load data in list with normalization
    inputList = [mean_normalization(np.load(os.path.join(mfccPath, fn))) for fn in os.listdir(mfccPath)]
    targetList = [np.load(os.path.join(labelPath, fn)) for fn in os.listdir(labelPath)]
    return preprocess(inputList,targetList)
"""