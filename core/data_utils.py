import os, sys
sys.path.append('../')
from random import randint
import numpy as np
from termcolor import cprint
from keras.utils import to_categorical
import scipy
import matplotlib.pyplot as plt
'''
if mode is fbank we assume that featLength is 40.
and if mode is mfcc we assume that featLength is 13.
'''
def compare_weight_similarity(teY,y_pred,label1,label2,label3=None,plot=True):
    print('compare with',str(label1),str(label2))
    from sklearn.metrics import mean_squared_error
    from termcolor import colored, cprint
    def norm(tmp):
        out= tmp / np.linalg.norm(tmp)
        return out
    #y_pred.shpae= [6835, 95, 36, 32] [B,Time,Freq,Channel]
    # lable number of 30 labels.
    LabelList = np.argmax(teY, 1)
    print(len(LabelList))
    assert(len(LabelList)==y_pred.shape[0])
    Array1 = np.zeros([y_pred.shape[1],y_pred.shape[2],y_pred.shape[3]])
    Array2 = np.zeros([y_pred.shape[1],y_pred.shape[2],y_pred.shape[3]])
    if label3: Array3 = np.zeros([y_pred.shape[1],y_pred.shape[2],y_pred.shape[3]]);count3=0
    count1,count2=0,0
    for index in range(LabelList.shape[0]):
        if LabelList[index] == label1:
            count1 +=1
            tmp = y_pred[index,:,:,:]
            tmp = norm(tmp)
            Array1 = np.add(Array1,tmp)
        elif LabelList[index] == label2:
            count2 += 1
            tmp = y_pred[index,:,:,:]
            tmp = norm(tmp)
            Array2 = np.add(Array2,tmp)
        elif label3 and LabelList[index] == label3:
            count3 += 1
            tmp = y_pred[index,:,:,:]
            tmp = norm(tmp)
            Array3 = np.add(Array3,tmp)
    Array1 = Array1 / count1
    Array2 = Array2 / count2
    if label3: Array3 = Array3 / count3
    sim1,sim2 = 0,0
    for i in range(y_pred.shape[3]):
        #sim += mean_squared_error(norm(Array1[:,:,i]), norm(Array2[:,:,i]))
        sim1 += mean_squared_error(Array1[:,:,i], Array3[:,:,i])
        sim2 += mean_squared_error(Array2[:,:,i], Array3[:,:,i])
        if plot:
            print(sim1,sim2)
            plt.subplot(3,1,1)
            plt.imshow(np.transpose(Array1[:,:,i]))
            plt.subplot(3,1,2)
            plt.imshow(np.transpose(Array2[:,:,i]))
            if label3: 
                plt.subplot(3,1,3)
                plt.imshow(np.transpose(Array3[:,:,i]))
            plt.show()
    sim1 /= y_pred.shape[3]
    sim2 /= y_pred.shape[3]
    ### Comparison pixel by pixel
    sim1,sim2 = 0,0
    for i in range(y_pred.shape[1]):
        for j in range(y_pred.shape[2]):
            sim1 += scipy.spatial.distance.cosine(Array1[i,j,:],Array3[i,j,:])
            sim2 += scipy.spatial.distance.cosine(Array2[i,j,:],Array3[i,j,:])
    cprint(str(sim1)+': comparison with '+str(label1) +' and '+str(label3),'blue')
    cprint(str(sim2)+': comparison with '+str(label2) +' and '+str(label3),'blue')

        #sim += skimage.measure.compare_mse(Array1[:,:,i], Array2[:,:,i])
    
    #sim = scipy.spatial.distance.cosine(Array1,Array2)
    return Array1, Array2, sim1
    
def select_feature(FeatureArray, feature_len):
    orig_feature_len = FeatureArray.shape[1]
    if orig_feature_len == 3*feature_len:
        return FeatureArray
    elif orig_feature_len > 3*feature_len:
        A = FeatureArray[:,:feature_len]
        B = FeatureArray[:,orig_feature_len/3:orig_feature_len/3+feature_len]
        C = FeatureArray[:,orig_feature_len/3*2:orig_feature_len/3*2+feature_len]
        result = np.concatenate((A,B,C),axis=1)
        return result
    else:
        raise ValueError('feature_len error')
    
def load_random_noisy_data(data_path, is_training, mode, feature_len, SNR=None):
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
        featLength = feature_len#40
    elif mode == 'mfcc':
        featLength = feature_len#13
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
        x[index, :,:] = select_feature(FeatArray,feature_len)
        y[index] = np.load(os.path.join(labelPath, filename))
    # [sample,99,120]
    x_new = np.zeros((x.shape[0],99,featLength,3))
    x_new[:,:,:,0] = x[:,:,:featLength]
    x_new[:,:,:,1] = x[:,:,featLength:2*featLength]
    x_new[:,:,:,2] = x[:,:,2*featLength:]
    y = to_categorical(y.astype('float32'))
    #y = np.expand_dims(y,axis=1)
    return x_new,y

def load_specific_noisy_data(saved_path, is_training, mode, feature_len, noise_name):
    mfccPath = os.path.join(saved_path,is_training,mode,noise_name)
    labelPath = os.path.join(saved_path,is_training,'label')

    if mode == 'fbank':
        featLength = feature_len#40
    elif mode == 'mfcc':
        featLength = feature_len#13
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
        x[index, :,:] = select_feature(FeatArray,feature_len)
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
        data = np.expand_dims(data[:,:,:,dimension],axis=3)
    elif dimension == 3:
        pass
    else:
        raise ValueError('Dimension value wrong')
    return data

def DATA(is_training, train_with, test_with, data_path, mode, feature_len, dimension):
    '''
    The first trX will have size [Number,99(time),40(feat),3]
    make it to channel 1 or 3
    '''
    if is_training == 'TRAIN':
        if train_with == 'clean':
            trX, trY = load_specific_noisy_data(data_path,is_training='TRAIN',mode=mode,feature_len=feature_len,noise_name='clean')
        elif train_with == 'noisy':
            raise ValueError('To be updated')
        else:
            raise ValueError('should set "--train_with" value')
        #
        if test_with == 'clean':
            vaX, vaY = load_specific_noisy_data(data_path,is_training='VALID',mode=mode,feature_len=feature_len,noise_name='clean')
        elif test_with == 'noisy':
            vaX, vaY = load_random_noisy_data(data_path,'VALID',mode=mode,feature_len=feature_len,)
        else:
            raise ValueError('should set "--test_with" value')
        #
        trX = Dimension(trX,dimension)
        vaX = Dimension(vaX,dimension)
        print(str(trX.shape),str(trY.shape),str(vaX.shape),str(vaY.shape))
        return (trX, trY, vaX, vaY)
    elif is_training =='TEST':
        teX, teY = load_specific_noisy_data(data_path,is_training='TEST',mode=mode,feature_len=feature_len,noise_name='clean') 
        # dimension start
        teX = Dimension(teX,dimension)
        print(str(teX.shape),str(teY.shape))
        return (teX, teY) #shape [sample, 120(f), 99?(t)], [sample,]


"""
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
"""



























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