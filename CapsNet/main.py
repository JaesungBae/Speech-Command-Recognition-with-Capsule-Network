###########################
# Capsule Network Main.py #
###########################

import tensorflow as tf
from tqdm import tqdm
import numpy as np
from termcolor import colored, cprint
import time, os, sys, scipy
sys.path.append('../')
# User
from core.data_utils import load_specific_noisy_data, load_random_noisy_data, DATA, Dimension, compare_weight_similarity
import core.data_utils as du
from core.args import parameter_print, args
from core.analysis import Analysis, TSNE_
from capsulenet import CapsNet_WithDecoder, margin_loss, CapsNet_NoDecoder, wieght_similarity
from core.data_utils import load_specific_noisy_data
# keras
from keras.utils import multi_gpu_model
from keras import callbacks, layers, optimizers
from keras import backend as K

noise_list = ['doing_the_dishes_SNR5','dude_miaowing_SNR5','exercise_bike_SNR5','pink_noise_SNR5','running_tap_SNR5','white_noise_SNR5']
def train(multi_model, data, save_path, args):
    trX, trY, vaX, vaY = data
    print(str(trX.shape),str(trY.shape),str(vaX.shape),str(vaY.shape))
    
    multi_model.compile(optimizer=optimizers.Adam(lr=args.learning_rate),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics=['accuracy'])    
    # callbacks
    log = callbacks.CSVLogger(save_path + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(save_path + '/weights-{epoch:03d}.h5py', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    earlystop = callbacks.EarlyStopping(monitor='val_decoder_acc', min_delta=0, patience=10, verbose=1, mode='auto')
    #lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.learning_rate * (0.9 ** args.num_epoch))
    #tb = callbacks.TensorBoard(log_dir=save_path + '/tensorboard-logs',
    #                           batch_size=args.batch_size, histogram_freq=args.debug)

    multi_model.fit([trX, trY],[trY,trX],
              batch_size=args.batch_size, epochs=args.num_epoch,
              #validation_split = 0.1,
              validation_data=[[vaX,vaY],[vaY,vaX]], 
              shuffle = True,
              callbacks=[log, checkpoint, earlystop])
    

def train_NoDecoder(multi_model, data, save_path, args):
    trX, trY, vaX, vaY = data
    print(str(trX.shape),str(trY.shape),str(vaX.shape),str(vaY.shape))
    
    if args.ex_name == 'best':
        multi_model.compile(optimizer=optimizers.SGD(lr=args.learning_rate),
                  loss=[margin_loss],
                  metrics=['accuracy'])    
    else:     
        multi_model.compile(optimizer=optimizers.Adam(lr=args.learning_rate),
                  loss=[margin_loss],
                  metrics=['accuracy'])    
    # callbacks
    log = callbacks.CSVLogger(save_path + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(save_path + '/weights-{epoch:03d}.h5py', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    earlystop = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
    #tb = callbacks.TensorBoard(log_dir=save_path + '/tensorboard-logs',
    #                           batch_size=args.batch_size, histogram_freq=args.debug)
    #lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.learning_rate * (0.9 ** args.num_epoch))

    multi_model.fit(trX, trY,
              batch_size=args.batch_size, epochs=args.num_epoch,
              #validation_split = 0.1,
              validation_data=[vaX,vaY], 
              shuffle = True,
              callbacks=[log, checkpoint, earlystop])


def test(model, data, args):
    start_time = time.time()
    teX,teY = data
    print('-'*20 + 'Begin: test with ' + '-'*20)
    y_pred = md.predict(teX,batch_size=args.batch_size)

    # Weight_similarity
    if args.weight_similarity:
        print(y_pred.shape)
        Array1,Array2, Sim = du.compare_weight_similarity(teY,y_pred,label1=0,label2=24,label3=25,plot=args.wsplot)
        cprint(Sim,'blue')

    # Test with 30 labels
    label30_acc = float(np.sum(np.argmax(y_pred, 1) == np.argmax(teY, 1)))/float(teY.shape[0])
    print('Test with 30 labels acc:', label30_acc )
    A = np.argmax(y_pred, 1)
    B = np.argmax(teY, 1)
    assert A.shape[0] == B.shape[0]
    du.pick_mis_recognized(B,A,label2=24,label3=25)
    
    # Test with 21 labels
    sub_label = [0,1,2,3,9,10,12,20,24,27]
    for i in range(A.shape[0]):
        if A[i] in sub_label: A[i] = 0
        if B[i] in sub_label: B[i] = 0
    label21_acc =  float(np.sum(A == B))/float(teY.shape[0])
    end_time = time.time()
    print('Test with 21 labels acc:', label21_acc)
    
    # Test with 10 labels
    sub_label = [0,1,2,3,5,6,7,9,10,12,13,17,19,20,21,23,24,25,27,29]
    for i in range(A.shape[0]):
        if A[i] in sub_label: A[i] = 0
        if B[i] in sub_label: B[i] = 0
    label10_acc =  float(np.sum(A == B))/float(teY.shape[0])
    print('Test with 10 labels acc:' + str(label10_acc))
    print('Time: ' + str(end_time-start_time))
    print('-'*20 + 'End: test' + '-'*20)
    return label30_acc, label21_acc


if __name__ == "__main__":
    args = args()
    ex_name = args.ex_name+'_'+args.train_with+'_'+args.test_with
    parameter_print(args,ex_name=ex_name,ModelType="CapsuleNet")
    save_path = os.path.join(args.project_path,'save',args.model,ex_name)
    cprint('save_path: '+str(save_path),'yellow')
    if args.is_training == 'TEST' and args.SNR == None:
        raise ValueError('For TEST you should set SNR')

    # Data load
    data = DATA(args.is_training, args.train_with, args.test_with,
                                args.data_path, feature_len=args.feature_len, mode=args.mode,dimension=args.dimension) #[sample,99,40,3]
    X,Y = data[0],data[1]

    # Define Model
    cprint(str(len(np.unique(np.argmax(Y, 1)))), 'red')
    with tf.device('/cpu:0'):
        if args.decoder == 1:
            model, eval_model, manipulate_model = CapsNet_WithDecoder(input_shape=X.shape[1:],
                                                          n_class=len(np.unique(np.argmax(Y, 1))),
                                                          kernel=args.kernel,
                                                          primary_channel=args.primary_channel,
                                                          primary_veclen=args.primary_veclen,
                                                          digit_veclen = args.digit_veclen,
                                                          dropout = args.dropout,
                                                          routings=args.routings,
                                                          decoderParm=(args.NumDecoderLayer,
                                                            [args.DecoderLayer1,args.DecoderLayer2,args.DecoderLayer3]
                                                            ),
                                                          model_size_info=args.model_size_info)
        else:
            model = CapsNet_NoDecoder(input_shape=X.shape[1:],
                                    n_class=len(np.unique(np.argmax(Y, 1))),
                                    kernel=args.kernel,
                                    primary_channel=args.primary_channel,
                                    primary_veclen=args.primary_veclen,
                                    digit_veclen = args.digit_veclen,
                                    dropout = args.dropout,
                                    routings=args.routings,
                                    model_size_info=args.model_size_info)
    if args.weight_similarity: model = wieght_similarity(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))),kernel=args.kernel)
    model.summary()
    multi_model = multi_gpu_model(model, gpus=args.gpus)

    # Save path and load model
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if args.keep and not args.weight_similarity:  # init the model weights with provided one
        cprint('load weight from:' + save_path + '/weights-%03d.h5py'%args.keep, 'yellow')
        multi_model.load_weights(save_path + '/weights-%03d.h5py'% args.keep)
        #model.load(save_path)
    elif args.keep and args.weight_similarity:
        cprint('weight_similarity','yellow')
        cprint('load weight from:' + save_path + '/weights-%03d.h5py'%args.keep, 'yellow')
        multi_model.load_weights(save_path + '/weights-%03d.h5py'% args.keep,by_name=True)
    else:
        if args.ex_name == 'best':
            cprint('load weight from:' + '/home/jsbae/STT2/KWS/save/CapsNet/0320_digitvec4_clean_clean/weights-012.h5py', 'yellow')
            multi_model.load_weights('/home/jsbae/STT2/KWS/save/CapsNet/0320_digitvec4_clean_clean/weights-012.h5py')
        else:
            cprint('save weight to: ' + save_path, 'yellow')

    # Model training or testing
    if args.is_training == 'TRAIN':
        if args.decoder == 1:
            train(multi_model,data=data,save_path=save_path,args=args)
        else:
            train_NoDecoder(multi_model,data=data,save_path=save_path,args=args)
        model.save_weights(save_path + '/trained_model.h5py')
        print('Trained model saved to \'%s/trained_model.h5py\'' % save_path)
    elif args.is_training == 'TEST':
        if args.keep == 0:
            raise ValueError('No weights are provided.')
        else:
            test_result = save_path + '/test_result.csv'
            fd_test_result = open(test_result,'a')
            fd_test_result.write('test on epoch '+str(args.keep)+'SNR'+str(args.SNR)+' dimension:'+str(args.dimension)+'\n')
            fd_test_result.write('test_mode,label30_acc,label21_acc\n')
            # clean test
            print('*'*30 + 'clean exp' + '*'*30)    
            label30_acc, label21_acc = test(multi_model, data=data,args=args)
            fd_test_result.write('clean,'+str(label30_acc)+','+str(label21_acc)+'\n')
            fd_test_result.flush()
            for i in range(6):
                print('*'*30 + 'Noisy '+ str(i+2) +' exp' + '*'*30)    
                if args.test_by=='noise':teX, teY = load_specific_noisy_data(args.data_path, 'TEST', args.mode, args.feature_len, noise_list[i]);cprint(noise_list[i],'red')
                elif args.test_by=='echo':teX, teY = load_specific_noisy_data(args.data_path, 'TEST', args.mode, args.feature_len, 'echo');cprint('echo','red')
                else: teX, teY = load_random_noisy_data(args.data_path,'TEST',args.mode, args.feature_len, SNR=args.SNR)
                teX = Dimension(teX,args.dimension)#teX = np.expand_dims(teX[:,:,:,1],axis=3)
                data = (teX, teY)
                label30_acc, label21_acc = test(multi_model, data=data,args=args)
                fd_test_result.write('noisy'+str(i)+','+str(label30_acc)+','+str(label21_acc)+'\n')
                fd_test_result.flush()
            fd_test_result.close()
    else:
        raise ValueError('Wrong is_training value')#'could not find %c in %s' % (ch,str)) 
# Code end
# For not decreasing issue: https://github.com/XifengGuo/CapsNet-Keras/issues/48




