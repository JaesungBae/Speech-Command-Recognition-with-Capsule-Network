import sys
sys.path.append('../')
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Activation, Dropout
from keras import models
from keras import initializers
from dense_tensor import DenseTensor, tensor_factorization_low_rank
#from keras.utils import np_utils

#[99,40,1]
def layer10_CNN(input_shape, n_class):
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

def CNN(input_shape, n_class):
	# Model: 0315_sameaspaper
    cnn_init = initializers.RandomNormal(stddev=0.01)
    model = Sequential()
    model.add(Conv2D(256, kernel_size=5, strides=(1,1),
        activation='relu', input_shape=input_shape,
        kernel_initializer=cnn_init))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    #model.add(Conv2D(128, (5,5), activation='relu'))
    model.add(Conv2D(256, (5,5), activation='relu',kernel_initializer=cnn_init))
    model.add(Conv2D(128, (5,5), activation='relu',kernel_initializer=cnn_init))
    model.add(Flatten())
    model.add(Dense(328, activation='relu'))
    model.add(Dense(192, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    return model

def jsbae_0314_clean_clean(input_shape, n_class):
    # Ref: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
    # Ref: https://github.com/flyyufelix/cnn_finetune/blob/master/vgg16.py
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

def lecun_0313_clean_clean_CNN(input_shape, n_class):
    # For TESt
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5,3), strides=(1,1),
        activation='relu', input_shape=input_shape))
    model.add(Flatten())
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
   