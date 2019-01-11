import tensorflow as tf
from tqdm import tqdm
import numpy as np
from termcolor import colored, cprint
import time 
import os, sys
sys.path.append('../')
# User
from core.data_utils import load_specific_noisy_data, load_random_noisy_data
from model import CNN
# Keras
from keras.utils import multi_gpu_model
from keras import callbacks, layers, optimizers
from keras import backend as K
from keras.losses import categorical_crossentropy


def parameter_print(args,ex_name):
    cprint('experiment name: '+ ex_name, 'cyan')
    cprint('batchsize: ' + str(args.batch_size), 'cyan')
    # cprint('num_feature: ' + str(args.num_feature), 'cyan')
    cprint('keep_prob: ' + str(args.keep_prob), 'cyan')
    #cprint('num_classes: ' + str(args.num_classes), 'cyan')
    cprint('learning_rate: ' + str(args.learning_rate), 'cyan')
    cprint('num_epoch: ' + str(args.num_epoch), 'cyan')

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
    #lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.learning_rate * (0.9 ** args.num_epoch))
    

    multi_model.fit(trX, trY,
              batch_size=args.batch_size, epochs=args.num_epoch,
              #validation_split = 0.1,
              validation_data=[vaX,vaY], 
              shuffle = True,
              callbacks=[log, checkpoint])

def test(model, data, args):
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


def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)



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
    parser.add_argument('--num_epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--gpus', default=2, type=int)
    # Should type
    parser.add_argument('-m','--model', default='CNN', type=str)
    parser.add_argument('-ex','--ex_name', default=None, type=str)
    parser.add_argument('-tr','--train_with', default=None,choices=['clean','noisy'], type=str)
    parser.add_argument('-te','--test_with', default=None,choices=['clean','noisy'], type=str)
    parser.add_argument('--keep', default=1,choices=[1,0], type=int)
    #
    parser.add_argument('--keep_prob', default=0.7, type=float)
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")

    args = parser.parse_args()
    ex_name = args.ex_name+'_'+args.train_with+'_'+args.test_with
    parameter_print(args,ex_name=ex_name)
    save_path = os.path.join(args.project_path,'save',args.model,ex_name)
    cprint('save_path: '+str(save_path),'yellow')

    # Setting & Parameters
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Data Load
    def DATA(train_with, test_with, data_path, mode):
        if train_with == 'clean':
            trX, trY = load_specific_noisy_data(data_path,is_training='TRAIN',mode=mode,noise_name='clean')
        elif train_with == 'noisy':
            trX, trY = load_random_noisy_data(data_path,'TRAIN',mode)
        else:
            raise ValueError('should set "--train_with" value')
        if test_with == 'clean':
            vaX, vaY = load_specific_noisy_data(data_path,is_training='VALID',mode=mode,noise_name='clean')
            teX, teY = load_specific_noisy_data(data_path,is_training='TEST',mode=mode,noise_name='clean')
        elif test_with == 'noisy':
            vaX, vaY = load_random_noisy_data(data_path,'VALID',mode)
            teX, teY = load_random_noisy_data(data_path,'TEST',mode)
        else:
            raise ValueError('should set "--test_with" value')
        return trX, trY, vaX, vaY, teX, teY #shape [sample, 99, 40, 3], [sample,1]
    StartTime = time.time()
    trX, trY, vaX, vaY, teX, teY = DATA(args.train_with, args.test_with, args.data_path,mode=args.mode)
    trX = np.expand_dims(trX[:,:,:,1],axis=3)
    vaX = np.expand_dims(vaX[:,:,:,1],axis=3)
    teX = np.expand_dims(teX[:,:,:,1],axis=3)
    EndTime = time.time()
    cprint('Takes '+str(EndTime-StartTime)+ 'time to load data','magenta')
    print(str(trX.shape),str(trY.shape),str(vaX.shape),str(vaY.shape))
    # Define Model
    #print(trY)
    #cprint(str(np.argmax(trY,1)),'red')
    cprint(str(len(np.unique(np.argmax(trY, 1)))), 'red')
    cprint(str(len(np.unique(np.argmax(vaY, 1)))), 'red')
    with tf.device('/cpu:0'):
        model = CNN(input_shape = trX.shape[1:],
            n_class=len(np.unique(np.argmax(trY, 1)))
            )
    model.summary()
    multi_model = multi_gpu_model(model, gpus=args.gpus)

    # Save path and load model
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if args.keep:  # init the model weights with provided one
        cprint('load weight from:' + save_path+'/weights-001.h5py', 'yellow')
        multi_model.load_weights(save_path+'/weights-001.h5py')
        #model.load(save_path+'/weights-001.h5py')
    else:
        cprint('save weight to: ' + save_path, 'yellow')

    # Model training or testing
    if args.is_training == 'TRAIN':
        train(multi_model,data=(trX,trY,vaX,vaY),save_path=save_path,args=args)

        model.save_weights(save_path + '/trained_model.h5py')
        print('Trained model saved to \'%s/trained_model.h5py\'' % save_path)

        test(model, data=(teX, teY),args=args)

    elif args.is_training == 'TEST':
        if args.keep == False:
            raise ValueError('No weights are provided.')
        else:
            manipulate_latent(manipulate_model, (teX,teY), args=args)
            test(eval_model, data=(teX, teY),args=args)
    else:
        raise ValueError('Wrong is_training value')#'could not find %c in %s' % (ch,str)) 
# Code end
# For not decreasing issue: https://github.com/XifengGuo/CapsNet-Keras/issues/48








def datetime():
    from datetime import datetime
    date = datetime.today().strftime('%Y%m%d_%H_%M')
    return date 

def data_load():
    x_train, y_packed = load_unbatched_data(args.data_path, args.is_training, args.level) # y_packed -> -2 is x_seqlen, -1 is y_seqlen
    x_valid, y_valid_packed = load_unbatched_data(args.data_path, 'VALID', args.level)

    x_train = np.expand_dims(x_train, axis=3) # inputX shape: [samples, time, freq , 1]
    x_train = np.reshape(x_train,[int(x_train.shape[0]), int(x_train.shape[1]), -1, 3]) # inputX shape: [batch, timestep, feature#, 3]
    x_valid = np.expand_dims(x_valid, axis=3) # inputX shape: [samples, time, freq , 1]
    x_valid = np.reshape(x_valid,[int(x_valid.shape[0]), int(x_valid.shape[1]), -1, 3]) # inputX shape: [batch, timestep, feature#, 3]
    
    padsec = x_train.shape[1] - x_valid.shape[1]
    x_valid = np.pad(x_valid, ((0,0),(0,padsec),(0,0),(0,0)),'constant',constant_values=0)

    padsec = y_packed.shape[1] - y_valid_packed.shape[1]
    pad_y = np.pad(y_valid_packed[:,:-2],((0,0),(0,padsec)), 'constant', constant_values=0)
    y_valid_packed = np.concatenate((pad_y, y_valid_packed[:,-2:]), axis=1)
    print(y_valid_packed.shape)
    # TEST
    x_test, y_test_packed = load_unbatched_data(args.data_path, args.is_training, args.level) # y_packed -> -2 is x_seqlen, -1 is y_seqlen
    x_test = np.expand_dims(x_test, axis=3) # inputX shape: [samples, time, freq , 1]
    x_test = np.reshape(x_test,[int(x_test.shape[0]), int(x_test.shape[1]), -1, 3])
    return x_train, y_packed, x_valid, y_valid_packed, x_test, y_test_packed