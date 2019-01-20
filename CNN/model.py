import sys
sys.path.append('../')
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Activation, Dropout, Input
from keras import layers
from keras import models
from keras import initializers
from termcolor import colored, cprint
#from dense_tensor import DenseTensor, tensor_factorization_low_rank
#from keras.utils import np_utils

#[99,40,1]
def wieght_similarity(input_shape, n_class, model_size_info):
    #CNN 0320_326464
    CNN1 = model_size_info[:5]
    CNN2 = model_size_info[5:10]
    L_size = model_size_info[-2]
    FC_size = model_size_info[-1]

    model = Sequential()
    model.add(Conv2D(CNN1[0], kernel_size=(CNN1[1],CNN1[2]), strides=(CNN1[3],CNN1[4]),
        activation='relu', input_shape=input_shape, padding='valid'))
    return model
def ref_cnn(input_shape, n_class, model_size_info):
    '''
    model_size_info: CNN1(channel, kernel, stride (time-freq)) + CNN2 + L + FC
    '''
    # model size
    CNN1 = model_size_info[:5]
    CNN2 = model_size_info[5:10]
    L_size = model_size_info[-2]
    FC_size = model_size_info[-1]
    # start
    cnn_init = initializers.glorot_normal()
    model = Sequential()
    model.add(Conv2D(CNN1[0], kernel_size=(CNN1[1],CNN1[2]), strides=(CNN1[3],CNN1[4]),
        activation='relu', input_shape=input_shape,
        kernel_initializer=cnn_init, padding='valid'))
    #model.add(layers.BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
    model.add(Conv2D(CNN2[0], kernel_size=(CNN2[1],CNN2[2]), strides=(CNN2[3],CNN2[4]), activation='relu', kernel_initializer=cnn_init, padding='valid'))
    #model.add(layers.BatchNormalization())
    #model.add(Dropout(0.5))
    # batch norm dropout
    #model.add(Conv2D(CNNChannel, (CNNkernel,CNNkernel), activation='relu',kernel_initializer=cnn_init, padding='valid'))
    # batch norm dropout
    model.add(Flatten())
    model.add(Dense(L_size, activation='relu'))
    model.add(Dense(FC_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    return model

def ref_2014icassp_dnn(input_shape, n_class, model_size_info):
    FC1, FC2, FC3 = model_size_info
    init = initializers.glorot_normal()
    model = Sequential()
    print(input_shape)
    model.add(layers.Reshape((input_shape[0]*input_shape[1]*input_shape[2],),input_shape=input_shape))
    #model.add(Flatten())
    model.add(Dense(FC1, activation='relu',kernel_initializer=init, bias_initializer='zeros'))
    model.add(Dense(FC2, activation='relu',kernel_initializer=init, bias_initializer='zeros'))
    model.add(Dense(FC3, activation='relu',kernel_initializer=init, bias_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax',kernel_initializer=init, bias_initializer='zeros'))
    return model

def ref_2015IS_cnn(input_shape, n_class, model_size_info):
    # time, freq
    cprint('****ref_2015IS_cnn*****','red')
    if len(model_size_info)>14:
        cnn1 = model_size_info[:7]#m,r,n,s,v,p,q
        cnn2 = model_size_info[7:14]#m,r,n,s,v,p,q
        dense_start = 14
        dense_len = len(model_size_info[14:])
        cnn_list = [cnn1,cnn2]
    else:
        cnn1 = model_size_info[:7]#m,r,n,s,v,p,q
        dense_start = 7
        dense_len = len(model_size_info[7:])
        cnn_list = [cnn1]

    cnn_init = initializers.glorot_normal()
    init = initializers.glorot_normal()
    model = Sequential()
    for index,cnn_info in enumerate(cnn_list):
        print(index,cnn_info)
        if index == 0:
            model.add(Conv2D(cnn_info[2], kernel_size=(cnn_info[0],cnn_info[1]), strides=(cnn_info[3],cnn_info[4]),
                activation='relu', input_shape=input_shape,
                kernel_initializer=cnn_init, padding='valid'))
        else:
            model.add(Conv2D(cnn_info[2], kernel_size=(cnn_info[0],cnn_info[1]), strides=(cnn_info[3],cnn_info[4]),
                activation='relu', kernel_initializer=cnn_init, padding='valid'))
        model.add(MaxPooling2D(pool_size=(cnn_info[5],cnn_info[6]), strides=(cnn_info[5],cnn_info[6])))
    model.add(Flatten())
    # Dense layer
    
    for i in range(dense_len):
        model.add(Dense(model_size_info[dense_start+i], activation='relu',kernel_initializer=init, bias_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax',kernel_initializer=init, bias_initializer='zeros'))
    return model

def ref_rnn(input_shape, n_class, model_size_info):
    '''
    model size info[-1]: LSTM for 0 and GRU for 1 
    '''
    cprint('**** ref_rnn ****','red')
    lstm1 = model_size_info[0]
    type_choice = model_size_info[-1]
    init = initializers.glorot_normal()
    # model
    model = Sequential()
    model.add(layers.Reshape((input_shape[0],input_shape[1],),input_shape=input_shape))
    if type_choice== 0: model.add(layers.LSTM(lstm1))
    elif type_choice== 1: model.add(layers.GRU(lstm1))
    else: raise ValueError('wrong type name')

    if len(model_size_info) == 3:
        lstm1, dnn1, type_choice = model_size_info
        model.add(Dense(dnn1, activation='relu',kernel_initializer=init, bias_initializer='zeros'))
        model.add(Dense(n_class, activation='softmax',kernel_initializer=init, bias_initializer='zeros'))
    elif len(model_size_info) == 2:
        lstm1,type_choice = model_size_info
        model.add(Dense(n_class, activation='softmax',kernel_initializer=init, bias_initializer='zeros'))
    else:
        raise ValueError('model size length too long.')
    return model

def ref_crnn(input_shape, n_class, model_size_info):
    cprint('**** CRNN ****','green')
    assert(len(model_size_info) == 9)
    cnn_info = model_size_info[:5]
    rnn_info = model_size_info[5:8]
    fc_unit = model_size_info[8]
    init = initializers.glorot_normal()
    # MODEL
    model = Sequential()
    model.add(Conv2D(cnn_info[0], kernel_size=(cnn_info[1],cnn_info[2]), strides=(cnn_info[3],cnn_info[4]),
                activation='relu', input_shape=input_shape,
                kernel_initializer=init, padding='valid'))
    model.add(layers.TimeDistributed(Flatten()))
    for i in range(rnn_info[0]-1):
        if rnn_info[2]== 0: model.add(layers.Bidirectional(layers.LSTM(rnn_info[1], return_sequences=True)))
        elif rnn_info[2]== 1: model.add(layers.Bidirectional(layers.GRU(rnn_info[1], return_sequences=True)))
        else: raise ValueError('wrong type name')
    if rnn_info[2]== 0: model.add(layers.Bidirectional(layers.LSTM(rnn_info[1])))
    elif rnn_info[2]== 1: model.add(layers.Bidirectional(layers.GRU(rnn_info[1])))
    else: raise ValueError('wrong type name')
    model.add(Dense(fc_unit, activation='relu',kernel_initializer=init, bias_initializer='zeros'))
    model.add(Dense(n_class, activation='softmax',kernel_initializer=init, bias_initializer='zeros'))
    return model

def CNN(input_shape, n_class, CNNkernel, CNNChannel,    DenseChannel):
    # ex_name: 0715
    cprint(str(CNNkernel)+str(CNNChannel)+str(DenseChannel),'yellow')
    cnn_init = initializers.glorot_normal()
    model = Sequential()
    model.add(Conv2D(CNNChannel, kernel_size=(CNNkernel,CNNkernel), strides=(1,1),
        activation='relu', input_shape=input_shape,
        kernel_initializer=cnn_init, padding='valid'))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    model.add(Conv2D(CNNChannel, (CNNkernel,CNNkernel), activation='relu', kernel_initializer=cnn_init, padding='valid'))
    model.add(Conv2D(CNNChannel, (CNNkernel,CNNkernel), activation='relu',kernel_initializer=cnn_init, padding='valid'))
    model.add(Flatten())
    model.add(Dense(DenseChannel, activation='relu'))
    model.add(Dense(DenseChannel//2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    return model
def OnlyDense(input_shape,n_class,DenseChannel):
    dense_init = initializers.glorot_normal()
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(2048, kernel_initializer=dense_init,input_shape=input_shape[0]*input_shape[1]))
    model.add(Dense(2048, kernel_initializer=dense_init))
    model.add(Dense(1024, kernel_initializer=dense_init))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    return model
    
def CNN___itworks(input_shape, n_class, CNNkernel, CNNChannel,    DenseChannel):
    # ex_name: 0715
    cnn_init = initializers.glorot_normal()
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(19,19), strides=(1,1),
        activation='relu', input_shape=input_shape,
        kernel_initializer=cnn_init, padding='valid'))
    #model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    model.add(Conv2D(32, (19,19), activation='relu', kernel_initializer=cnn_init, padding='valid'))
    #model.add(Conv2D(32, (5,5), activation='relu',kernel_initializer=cnn_init, padding='valid'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    return model

def CNN_(input_shape, n_class):
    cnn_init = initializers.glorot_normal()
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1),
        activation='relu', input_shape=input_shape,
        kernel_initializer=cnn_init, padding='valid'))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    model.add(Conv2D(32, (5,5), activation='relu', kernel_initializer=cnn_init, padding='valid'))
    model.add(Conv2D(32, (5,5), activation='relu',kernel_initializer=cnn_init, padding='valid'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    return model

def CNN_0314(input_shape, n_class):
    cnn_init = initializers.glorot_normal()
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5,5), strides=(1,1),
        activation='relu', input_shape=input_shape,
        kernel_initializer=cnn_init, padding='valid'))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    model.add(Conv2D(256, (5,5), activation='relu', kernel_initializer=cnn_init, padding='valid'))
    model.add(Conv2D(256, (5,5), activation='relu',kernel_initializer=cnn_init, padding='valid'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    return model

def CNN_0320_326464(input_shape, n_class):
    cnn_init = initializers.glorot_normal()
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1),
        activation='relu', input_shape=input_shape,
        kernel_initializer=cnn_init, padding='valid'))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    model.add(Conv2D(64, (5,5), activation='relu', kernel_initializer=cnn_init, padding='valid'))
    model.add(Conv2D(64, (7,7), activation='relu',kernel_initializer=cnn_init, padding='valid'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    return model

def CNN_0320(input_shape, n_class):
    cnn_init = initializers.glorot_normal()
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5,5), strides=(1,1),
        activation='relu', input_shape=input_shape,
        kernel_initializer=cnn_init, padding='valid'))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    model.add(Conv2D(64, (5,5), activation='relu', kernel_initializer=cnn_init, padding='valid'))
    model.add(Conv2D(64, (5,5), activation='relu',kernel_initializer=cnn_init, padding='valid'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    return model


































def sch_0320_64128128(input_shape, n_class):
    cnn_init = initializers.glorot_normal()
    model = Sequential()
    model.add(Conv2D(32, kernel_size=5, strides=(1,1),
        activation='relu', input_shape=input_shape,
        kernel_initializer=cnn_init, padding='valid'))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    model.add(Conv2D(64, (5,5), activation='relu', kernel_initializer=cnn_init, padding='valid'))
    model.add(Conv2D(64, (5,5), activation='relu',kernel_initializer=cnn_init, padding='valid'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    return model

def jsbae_0318_C2F1(input_shape, n_class):
    #jsbae_0314_clean_clean ## BEST MODEL UNTIL NOW
    cnn_init = initializers.glorot_normal()
    model = Sequential()
    model.add(Conv2D(128, kernel_size=5, strides=(1,1),
        activation='relu', input_shape=input_shape,
        kernel_initializer=cnn_init, padding='valid'))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    #model.add(Conv2D(128, (5,5), activation='relu'))
    model.add(Conv2D(256, (5,5), activation='relu', kernel_initializer=cnn_init, padding='valid'))
    #model.add(Conv2D(256, (5,5), activation='relu',kernel_initializer=cnn_init, padding='valid'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    #model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    return model

def jsbae_0314_clean_clean(input_shape, n_class):
    #jsbae_0314_clean_clean ## BEST MODEL UNTIL NOW
    cnn_init = initializers.RandomNormal(stddev=0.01)
    model = Sequential()
    model.add(Conv2D(128, kernel_size=5, strides=(1,1),
        activation='relu', input_shape=input_shape,
        kernel_initializer=cnn_init))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    #model.add(Conv2D(128, (5,5), activation='relu'))
    model.add(Conv2D(256, (5,5), activation='relu',kernel_initializer=cnn_init))
    model.add(Conv2D(256, (5,5), activation='relu',kernel_initializer=cnn_init))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    return model

def FAIL_lecun_0313_clean_clean_CNN(input_shape, n_class):
    # For TESt
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5,3), strides=(1,1),
        activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(n_class, activation='softmax'))
    return model

def FAIL_jsbae_0315_Desne_3(input_shape, n_class):
    # 0315_Dense 3 layer
    cnn_init = initializers.RandomNormal(stddev=0.01)
    model = Sequential()
    model.add(Conv2D(128, kernel_size=5, strides=(1,1),
        activation='relu', input_shape=input_shape,
        kernel_initializer=cnn_init))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    #model.add(Conv2D(128, (5,5), activation='relu'))
    model.add(Conv2D(256, (5,5), activation='relu',kernel_initializer=cnn_init))
    model.add(Conv2D(256, (5,5), activation='relu',kernel_initializer=cnn_init))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    return model

def FAIL_layer10_CNN(input_shape, n_class):
    model = Sequential() 
    model.add(Conv2D(128, kernel_size=(5,3), strides=(1, 1),
                     activation='relu',
                     padding='same',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    model.add(Conv2D(128, (5,3), padding='same',activation='relu'))
    model.add(Conv2D(128, (5,3), padding='same',activation='relu'))
    model.add(Conv2D(128, (5,3), padding='same',activation='relu'))
    model.add(Conv2D(256, (5,3), padding='same',activation='relu'))
    model.add(Conv2D(256, (5,3), padding='same',activation='relu'))
    model.add(Conv2D(256, (5,3), padding='same',activation='relu'))
    model.add(Conv2D(256, (5,3), padding='same',activation='relu'))
    model.add(Conv2D(256, (5,3), padding='same',activation='relu'))
    model.add(Conv2D(256, (5,3), padding='same',activation='relu'))
    #model.add(Conv2D(128, (5, 3), activation='relu'))
    #model.add(Conv2D(256, (9, 9), padding='same',activation='relu'))
    #model.add(Conv2D(256, (9, 9), padding='same',activation='relu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    #model.add(Dense(1024, activation='relu'))
    #model.add(Dense(1024, activation='relu'))
    model.add(Dense(n_class, activation='softmax'))
    return model

'''
## Try Low Rank
def CNN(input_shape, n_class):
    x = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(20,8), 
        strides=1, padding='valid', activation='relu', name='conv1')(x)
    conv1_pool = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(10,4), 
        strides=1, padding='valid', activation='relu', name='conv1')(conv1_pool)
    factorization = tensor_factorization_low_rank(q=32)
    lin = DenseTensor(units=128,
                    activation=None,
                    kernel_regularizer=None,
                    factorization=factorization)(conv2)
    out = Dense(n_class,activation='softmax')(lin)

    model = models.Model(x,out)
    return model

'''
   