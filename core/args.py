import argparse
from termcolor import colored, cprint

def args():
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
    #parser.add_argument('--data_path', default= '/home/jsbae/STT/From_server_ASr/KWS_feature_saved', type=str)
    parser.add_argument('--project_path', default='/home/jsbae/STT2/SCR_INTERSPEECH2018', type=str)
    # Parameters
    parser.add_argument('-lr','--learning_rate', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gpus', default=2, type=int)
    parser.add_argument('--SNR', default=None, type=int)
    # Should type
    parser.add_argument('-m','--model', default=None, type=str)
    parser.add_argument('-ex','--ex_name', default=None, type=str)
    parser.add_argument('-tr','--train_with', default=None,choices=['clean','noisy'], type=str)
    parser.add_argument('-te','--test_with', default=None,choices=['clean','noisy'], type=str)
    parser.add_argument('-fte','--final_test_with', default=None,choices=['clean','noisy'], type=str)
    # Added
    parser.add_argument('--dimension', default=3, choices=[0,1,2,3],type=int)
    parser.add_argument('-d','--decoder', default=None,choices=[0,1], type=int)
    parser.add_argument('--keep', default=None, type=int)
    #
    parser.add_argument('--keep_prob', default=0.7, type=float)
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    # CNN experiment
    parser.add_argument('-CNNk','--CNNkernel', default=5, type=int, help="CNN 3 layer's kernel size.")
    parser.add_argument('-CNNC','--CNNChannel', default=32, type=int, help="CNN 3 layer's channel size.")
    parser.add_argument('-Dense','--DenseChannel', default=128, type=int, help="Dense 2 layer's channel size.")
    # Capsule Experiment
    parser.add_argument('--kernel', default=19, type=int,
                        help="Convlution and Primary cpasule layer's kernel size.")
    parser.add_argument('--primary_channel', default=32, type=int)
    parser.add_argument('--primary_veclen', default=4, type=int)
    parser.add_argument('--digit_veclen', default=16, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    # Capsule Deocoder Experiment
    parser.add_argument('-ND','--NumDecoderLayer', default=3, type=int)
    parser.add_argument('-DL1','--DecoderLayer1', default=512, type=int)
    parser.add_argument('-DL2','--DecoderLayer2', default=1024, type=int)
    parser.add_argument('-DL3','--DecoderLayer3', default=2048, type=int)
    args = parser.parse_args()
    return args

def parameter_print(args,ex_name,ModelType):
    cprint('experiment name: '+ ex_name, 'cyan')
    cprint('batchsize: ' + str(args.batch_size), 'cyan')
    cprint('keep_prob: ' + str(args.keep_prob), 'cyan')
    cprint('learning_rate: ' + str(args.learning_rate), 'cyan')
    cprint('num_epoch: ' + str(args.num_epoch), 'cyan')
    cprint('*'*10 + str(ModelType) + '*'*10, 'cyan')
    if ModelType=='CNN':
        cprint('CNNk: ' + str(args.CNNkernel), 'cyan')
        cprint('CNNC: ' + str(args.CNNChannel), 'cyan')
        cprint('Dense: ' + str(args.DenseChannel), 'cyan')
    else:
        cprint('kernel size: '+ str(args.kernel),'cyan')
        cprint('primary_channel: '+ str(args.primary_channel),'cyan')
        cprint('Decoder: '+ str(args.decoder),'cyan')
        cprint('primary_veclen: '+ str(args.primary_veclen),'cyan')
        cprint('digit_veclen: '+ str(args.digit_veclen),'cyan')