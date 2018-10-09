import sys
sys.path.append('../')
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Activation, Dropout
from keras import models
from keras import initializers
#from dense_tensor import DenseTensor, tensor_factorization_low_rank
#from keras.utils import np_utils

#[99,40,1]
def CNN(input_shape, n_class, CNNkernel, CNNChannel,    DenseChannel):
    # ex_name: 0715
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
   