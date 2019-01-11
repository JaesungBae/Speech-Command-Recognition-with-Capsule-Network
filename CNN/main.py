###############
# CNN Main.py #
###############

import tensorflow as tf
from tqdm import tqdm
import numpy as np
from termcolor import colored, cprint
import time 
import os, sys
sys.path.append('../')
# User
from core.data_utils import load_specific_noisy_data, load_random_noisy_data, DATA, Dimension, compare_weight_similarity
from core.args import parameter_print
#from core.analysis import Analysis
from model import CNN, CNN_0320, CNN_0314, CNN_0320_326464, OnlyDense, wieght_similarity, ref_cnn
import model
# Keras
from keras.utils import multi_gpu_model
from keras import callbacks, layers, optimizers
from keras import backend as K
from keras.losses import categorical_crossentropy
import scipy
noise_list = ['doing_the_dishes_SNR5','dude_miaowing_SNR5','exercise_bike_SNR5','pink_noise_SNR5','running_tap_SNR5','white_noise_SNR5']
def train(multi_model, data, save_path, args):
    trX, trY, vaX, vaY = data
    print(str(trX.shape),str(trY.shape),str(vaX.shape),str(vaY.shape))
    
    multi_model.compile(optimizer=optimizers.Adam(lr=args.learning_rate),
                  loss=categorical_crossentropy,
                  metrics=['accuracy']
                  )
    # callbacks
    log = callbacks.CSVLogger(save_path + '/log.csv')
    #tb = callbacks.TensorBoard(log_dir=save_path + '/tensorboard-logs',
    #                           batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(save_path + '/weights-{epoch:03d}.h5py', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    earlystop = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto')
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
    print('-'*30 + 'Begin: test with ' + '-'*30)
    y_pred = model.predict(teX,batch_size=args.batch_size)
    #
    if args.weight_similarity:
        print(y_pred.shape)
        Array1,Array2, Sim = compare_weight_similarity(teY,y_pred,label1=0,label2=24,label3=25,plot=False)
        cprint(Sim,'blue')
    #
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
    cprint('Time: '+str(time.time()-start_time),'yellow')
    # Check 10 label
    sub_label = [0,1,2,3,5,6,7,9,10,12,13,17,19,20,21,23,24,25,27,29]
    for i in range(A.shape[0]):
        if A[i] in sub_label:
            A[i] = 0
        if B[i] in sub_label:
            B[i] = 0
    label10_acc =  float(np.sum(A == B))/float(teY.shape[0])
    cprint('Test with 10 labels acc:' + str(label10_acc),'red')
    '''
    ##
    ##Confusion Matrix Generate
    ##
    Anal = Analysis(args)
    #Anal.ConfusionMatrix_Generate(y_pred30=np.argmax(y_pred, 1), teY30=np.argmax(teY, 1), y_pred20=A,teY20=B)
    #Anal.RawPredictMatrix_Generate(np.reshape(np.argmax(y_pred, 1), (-1,1)), teY)
    Anal.MatrixSave(np.reshape(np.argmax(y_pred, 1), (-1,1)), teY)
    Anal.StopCode()
    '''
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
    from core.data_utils import load_specific_noisy_data
    from core.args import args
    args = args()
    '''
    parser.add_argument('--kernel', default=19, type=int,
                        help="Convlution and Primary cpasule layer's kernel size.")
    parser.add_argument('--primary_channel', default=32, type=int)
    parser.add_argument('--primary_veclen', default=8, type=int)
    '''
    ex_name = args.ex_name+'_'+args.train_with+'_'+args.test_with
    parameter_print(args,ex_name=ex_name,ModelType='CNN')
    save_path = os.path.join(args.project_path,'save',args.model,ex_name)
    cprint('save_path: '+str(save_path),'yellow')
    if args.is_training == 'TEST' and args.SNR == None:
        raise ValueError('For TEST you should set SNR')
    # Setting & Parameters
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Data Load
    data = DATA(args.is_training, args.train_with, args.test_with,
                                args.data_path, mode=args.mode, feature_len=args.feature_len,dimension=args.dimension) #[sample,99,40,3]
    X = data[0]
    Y = data[1]
    # Define Model
    cprint(str(len(np.unique(np.argmax(Y, 1)))), 'red')
    with tf.device('/cpu:0'):
        cprint('+'*10,'yellow')
        print(args.ex_name)
        if args.ex_name == '0314':
            cprint('+'*10,'yellow')
            model = CNN_0314(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))),)
        elif args.ex_name == '0320':
            cprint('+'*10,'yellow')
            model = CNN_0320(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))),)
        elif args.ex_name == '0320_326464' or args.ex_name =='dim0_0929':
            cprint('+'*10,'yellow')
            model = CNN_0320_326464(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))),)
        elif 'onlydense' in args.ex_name.lower():
        	cprint('Only Dense Experience','yellow')
        	model = OnlyDense(input_shape=X.shape[1:],n_class=len(np.unique(np.argmax(Y, 1))),DenseChannel=args.DenseChannel)
        elif 'ref_cnn' in args.ex_name.lower():
            model = ref_cnn(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))), model_size_info=args.model_size_info)
        elif 'ref_2014icassp_dnn' in args.ex_name.lower():
        	model = model.ref_2014icassp_dnn(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))), model_size_info=args.model_size_info)
        elif 'ref_2015is_cnn' in args.ex_name.lower():
            model = model.ref_2015IS_cnn(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))), model_size_info=args.model_size_info)
        elif 'ref_rnn' in args.ex_name.lower():
            model = model.ref_rnn(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))), model_size_info=args.model_size_info)
        elif 'ref_crnn' in args.ex_name.lower():
        	model = model.ref_crnn(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))), model_size_info=args.model_size_info)
        else:
            cprint('go to else loop','red')
            model = CNN(input_shape=X.shape[1:],
                n_class=len(np.unique(np.argmax(Y, 1))),
                CNNkernel=args.CNNkernel,
                CNNChannel=args.CNNChannel,
                DenseChannel=args.DenseChannel,
                )
    if args.weight_similarity: model = wieght_similarity(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))))
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
        cprint('save weight to: ' + save_path, 'yellow')

    # Model training or testing
    if args.is_training == 'TRAIN':
        train(multi_model,data=data,save_path=save_path,args=args)
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
            label30_acc, label21_acc = test(multi_model, data=data,args=args)
            fd_test_result.write('clean,'+str(label30_acc)+','+str(label21_acc)+'\n')
            fd_test_result.flush()
            for i in range(6):
                print('*'*30 + 'Noisy '+ str(i+2) +' exp' + '*'*30)    
                if args.test_by=='noise':teX, teY = load_specific_noisy_data(args.data_path, 'TEST', args.mode, args.feature_len, noise_list[i]);cprint(noise_list[i],'red')
                elif args.test_by=='echo':teX, teY = load_specific_noisy_data(args.data_path, 'TEST',  args.mode,args.feature_len, 'echo');cprint('echo','red')
                else:teX, teY = load_random_noisy_data(args.data_path,'TEST',args.mode, args.feature_len, SNR=args.SNR)              
                #teX = np.expand_dims(teX[:,:,:,1],axis=3)
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