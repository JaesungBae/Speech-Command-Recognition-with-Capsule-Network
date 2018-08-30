import tensorflow as tf
from tqdm import tqdm
import numpy as np
from termcolor import colored, cprint
import time 
import os, sys
sys.path.append('../')
# User
from core.data_utils import load_specific_noisy_data, load_random_noisy_data
from capsulenet import CapsNet, margin_loss, CapsNet_NoDecoder
# keras
from keras.utils import multi_gpu_model
from keras import callbacks, layers, optimizers
from keras import backend as K


def parameter_print(args,ex_name):
    cprint('experiment name: '+ ex_name, 'cyan')
    cprint('batchsize: ' + str(args.batch_size), 'cyan')
    # cprint('num_feature: ' + str(args.num_feature), 'cyan')
    cprint('keep_prob: ' + str(args.keep_prob), 'cyan')
    #cprint('num_classes: ' + str(args.num_classes), 'cyan')
    cprint('learning_rate: ' + str(args.learning_rate), 'cyan')
    cprint('num_epoch: ' + str(args.num_epoch), 'cyan')
    cprint('*'*10, 'cyan')
    cprint('kernel size: '+ str(args.kernel),'cyan')
    cprint('primary_channel: '+ str(args.primary_channel),'cyan')
    cprint('Decoder: '+ str(args.decoder),'cyan')
    cprint('primary_veclen: '+ str(args.primary_veclen),'cyan')
    cprint('digit_veclen: '+ str(args.digit_veclen),'cyan')

def train(multi_model, data, save_path, args):
    trX, trY, vaX, vaY = data
    print(str(trX.shape),str(trY.shape),str(vaX.shape),str(vaY.shape))
    
    multi_model.compile(optimizer=optimizers.Adam(lr=args.learning_rate),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics=['accuracy'])    
    # callbacks
    log = callbacks.CSVLogger(save_path + '/log.csv')
    #tb = callbacks.TensorBoard(log_dir=save_path + '/tensorboard-logs',
    #                           batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(save_path + '/weights-{epoch:03d}.h5py', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    #lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.learning_rate * (0.9 ** args.num_epoch))
    multi_model.fit([trX, trY],[trY,trX],
              batch_size=args.batch_size, epochs=args.num_epoch,
              #validation_split = 0.1,
              validation_data=[[vaX,vaY],[vaY,vaX]], 
              shuffle = True,
              callbacks=[log, checkpoint])
    
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
    #tb = callbacks.TensorBoard(log_dir=save_path + '/tensorboard-logs',
    #                           batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(save_path + '/weights-{epoch:03d}.h5py', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    earlystop = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
    #lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.learning_rate * (0.9 ** args.num_epoch))
    multi_model.fit(trX, trY,
              batch_size=args.batch_size, epochs=args.num_epoch,
              #validation_split = 0.1,
              validation_data=[vaX,vaY], 
              shuffle = True,
              callbacks=[log, checkpoint, earlystop])

def test(model, data, args):
    teX,teY = data
    print('-'*30 + 'Begin: test with ' + '-'*30)
    if args.decoder == 1:
        y_pred, x_recon = model.predict([teX,teY],batch_size=args.batch_size)
    else:
        y_pred= model.predict(teX,batch_size=args.batch_size)
    label30_acc = float(np.sum(np.argmax(y_pred, 1) == np.argmax(teY, 1)))/float(teY.shape[0])
    print('Test with 30 labels acc:', label30_acc )
    A = np.argmax(y_pred, 1)
    B = np.argmax(teY, 1)
    assert A.shape[0] == B.shape[0]
    print(A.shape, A.shape[0], B.shape, B.shape[0])
    sub_label = [0,1,2,3,9,10,12,20,24,27]
    
    for i in range(A.shape[0]):
        if A[i] in sub_label:
            A[i] = 0
        if B[i] in sub_label:
            B[i] = 0
    label21_acc =  float(np.sum(A == B))/float(teY.shape[0])
    print('Test with 20 labels acc:', label21_acc)
    print('-'*30 + 'End: test' + '-'*30)
    return label30_acc, label21_acc
    '''
    def text_to_label(text):
    return {'bed':0, 'bird':1, 'cat':2, 'dog':3, 'down':4, 
        'eight':5, 'five':6, 'four':7, 'go':8, 'happy':9, 
        'house':10, 'left':11, 'marvin':12, 'nine':13, 'no':14, 
        'off':15, 'on':16, 'one':17, 'right':18, 'seven':19, 
        'sheila':20, 'six':21, 'stop':22, 'three':23, 'tree':24,
        'two':25, 'up':26, 'wow':27, 'yes':28, 'zero':29}.get(text, 30)
    '''
    '''
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()
    '''



if __name__ == "__main__":
    import argparse
    from core.data_utils import load_specific_noisy_data
    parser = argparse.ArgumentParser(description="Preprocessing for Google Speech Command dataset.")
    parser.add_argument('--mode', default='fbank',choices=['mfcc','fbank'], type=str)
    parser.add_argument('--feature_len', default=40, type=int)
    parser.add_argument('--noise_name', default='clean', 
        choices=['clean','exercise_bike','pink_noise','doing_the_dishes','running_tap','dude_miaowing','white_noise'],
        type=str)
    parser.add_argument('--is_training', default='TRAIN',choices=['TRAIN','TEST'], type=str)
    #Experiment
    #Path
    parser.add_argument('--data_path', default= '/DATA/jsbae/KWS_feature_saved', type=str)
    parser.add_argument('--project_path', default='/home/jsbae/STT2/KWS', type=str)
    # Parameters
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gpus', default=2, type=int)
    parser.add_argument('--SNR', default=None, type=int)
    # Should type
    parser.add_argument('-m','--model', default=None, type=str)
    parser.add_argument('-ex','--ex_name', default=None, type=str)
    parser.add_argument('-tr','--train_with', default=None,choices=['clean','noisy'], type=str)
    parser.add_argument('-te','--test_with', default=None,choices=['clean','noisy'], type=str)
    parser.add_argument('-fte','--final_test_with', default=None,choices=['clean','noisy'], type=str)
    parser.add_argument('-d','--decoder', default=None,choices=[0,1], type=int)
    parser.add_argument('--keep', default=None, type=int)
    #
    parser.add_argument('--keep_prob', default=0.7, type=float)
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    #
    parser.add_argument('--kernel', default=19, type=int,
                        help="Convlution and Primary cpasule layer's kernel size.")
    parser.add_argument('--primary_channel', default=32, type=int)
    parser.add_argument('--primary_veclen', default=4, type=int)
    parser.add_argument('--digit_veclen', default=16, type=int)
    parser.add_argument('--dropout', default=0, type=float)

    args = parser.parse_args()
    ex_name = args.ex_name+'_'+args.train_with+'_'+args.test_with
    parameter_print(args,ex_name=ex_name)
    save_path = os.path.join(args.project_path,'save',args.model,ex_name)
    cprint('save_path: '+str(save_path),'yellow')
    if args.is_training == 'TEST' and args.SNR == None:
        raise ValueError('For TEST you should set SNR')

    # Setting & Parameters
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Data Load
    def DATA(is_training, train_with, test_with, data_path, mode):
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
            trX = np.expand_dims(trX[:,:,:,1],axis=3)
            vaX = np.expand_dims(vaX[:,:,:,1],axis=3)
            print(str(trX.shape),str(trY.shape),str(vaX.shape),str(vaY.shape))
            return (trX, trY, vaX, vaY)
        elif is_training =='TEST':
            teX, teY = load_specific_noisy_data(data_path,is_training='TEST',mode=mode,noise_name='clean') 
            teX = np.expand_dims(teX[:,:,:,1],axis=3)
            print(str(teX.shape),str(teY.shape))
            return (teX, teY) #shape [sample, 120(f), 99?(t)], [sample,]

    data = DATA(args.is_training, args.train_with, args.test_with,
                                args.data_path, mode=args.mode) #[sample,99,40,3]
    X = data[0]
    Y = data[1]
    # Define Model
    cprint(str(len(np.unique(np.argmax(Y, 1)))), 'red')
    with tf.device('/cpu:0'):
        if args.decoder == 1:
            model, eval_model, manipulate_model = CapsNet(input_shape=X.shape[1:],
                                                          n_class=len(np.unique(np.argmax(Y, 1))),
                                                          kernel=args.kernel,
                                                          primary_channel=args.primary_channel,
                                                          primary_veclen=args.primary_veclen,
                                                          digit_veclen = args.digit_veclen,
                                                          dropout = args.dropout,
                                                          routings=args.routings)
        else:
            model = CapsNet_NoDecoder(input_shape=X.shape[1:],
                                    n_class=len(np.unique(np.argmax(Y, 1))),
                                    kernel=args.kernel,
                                    primary_channel=args.primary_channel,
                                    primary_veclen=args.primary_veclen,
                                    digit_veclen = args.digit_veclen,
                                    dropout = args.dropout,
                                    routings=args.routings)
    model.summary()
    multi_model = multi_gpu_model(model, gpus=args.gpus)

    # Save path and load model
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if args.keep:  # init the model weights with provided one
        cprint('load weight from:' + save_path + '/weights-%03d.h5py'%args.keep, 'yellow')
        multi_model.load_weights(save_path + '/weights-%03d.h5py'% args.keep)
        #model.load(save_path)
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
            fd_test_result.write('test on epoch '+str(args.keep)+'SNR'+str(args.SNR)+'\n')
            fd_test_result.write('test_mode,label30_acc,label21_acc\n')
            # clean test
            print('*'*30 + 'clean exp' + '*'*30)    
            label30_acc, label21_acc = test(multi_model, data=data,args=args)
            fd_test_result.write('clean,'+str(label30_acc)+','+str(label21_acc)+'\n')
            fd_test_result.flush()
            for i in range(5):
                print('*'*30 + 'Noisy '+ str(i+2) +' exp' + '*'*30)    
                teX, teY = load_random_noisy_data(args.data_path,'TEST',args.mode,SNR=args.SNR)
                teX = np.expand_dims(teX[:,:,:,1],axis=3)
                data = (teX, teY)
                label30_acc, label21_acc = test(multi_model, data=data,args=args)
                fd_test_result.write('noisy'+str(i)+','+str(label30_acc)+','+str(label21_acc)+'\n')
                fd_test_result.flush()
            fd_test_result.close()
    else:
        raise ValueError('Wrong is_training value')#'could not find %c in %s' % (ch,str)) 
# Code end
# For not decreasing issue: https://github.com/XifengGuo/CapsNet-Keras/issues/48




