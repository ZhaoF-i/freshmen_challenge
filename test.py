import argparse
import os
import yaml
import soundfile as sf
import logging as log
import numpy as np
from pathlib import Path
from torch import nn
from metrics import Metrics
from networks.TCN import NET_Wrapper
from utils.Checkpoint import Checkpoint
from utils.progressbar import progressbar as pb
from utils.stft_istft import STFT
from utils.util import makedirs, gen_list

import torch
from torch.nn.utils.rnn import *
from torch.autograd.variable import *
import requests


class Test(object):
    def __init__(self, inpath, outpath, type='online', suffix='mix.wav'):
        self.inpath = inpath
        self.outpath = outpath
        self.type = type
        self.suffix = suffix
        self.STFT = STFT(config['WIN_LEN'], config['WIN_OFFSET']).cuda()

    def forward(self, network):
        network.eval()
        tt_lst = gen_list(self.inpath, self.suffix)
        tt_len = len(tt_lst)
        pbar = pb(0, tt_len)
        pbar.start()
        for i in range(tt_len):
            pbar.update_progress(i, 'tt', '')
            mix, fs = sf.read(self.inpath + '/' + tt_lst[i])
            mixture = Variable(torch.FloatTensor(mix.astype('float32')))
            len_speech = len(mixture)
            alpha = 1 / torch.sqrt(torch.sum(mixture ** 2) / len_speech)
            mixture = mixture * alpha
            mixture = mixture.reshape([1, -1])

            """------------------------------------modify  area------------------------------------"""
            with torch.no_grad():
                est = network(mixture)
            real = est[0] * torch.cos(est[2].permute(0, 2, 1))
            imag = est[0] * torch.sin(est[2].permute(0, 2, 1))
            est_speech = self.STFT.inverse(torch.stack([real, imag], 3))
            est = est_speech[0].data.cpu().numpy()
            """------------------------------------modify  area------------------------------------"""
            if self.type == 'online':
                clean_name = tt_lst[i][:-7] + 'clean.wav'
                clean, _ = sf.read(self.inpath + '/' + clean_name)
                sf.write(self.outpath + tt_lst[i][:-len(self.suffix) - 1] + '_clean.wav', clean[:est.size], fs)
            sf.write(self.outpath + tt_lst[i][:-len(self.suffix) - 1] + '_est.wav', est / alpha, fs)
            sf.write(self.outpath + tt_lst[i][:-len(self.suffix) - 1] + '_mix.wav', mix[:est.size], fs)

        pbar.finish()


if __name__ == '__main__':
    """
        environment part
        """
    # loading argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="trained model name, retrain if no input", default='none')
    parser.add_argument("-y", "--yaml_name", help="config file name")
    parser.add_argument("-n", "--new_test",
                        help="generate new test date[1] or only compute metrics with exist data[0]",default=0,type=int)
    args = parser.parse_args()

    # loading config
    _abspath = Path(os.path.abspath(__file__)).parent
    _project = _abspath.stem
    _yaml_path = os.path.join(_abspath, 'configs/' + args.yaml_name)
    with open(_yaml_path, 'r') as f_yaml:
        config = yaml.load(f_yaml, Loader=yaml.FullLoader)

    # if online test
    _outpath = config['OUTPUT_DIR'] + _project + config['WORKSPACE']
    # if offline test
    # _outpath = config['OFFLINE_TEST_DIR'] + _project + config['WORKSPACE']
    outpath = _outpath + '/estimations/'
    makedirs([outpath])

    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_ID']
    if args.new_test:
        network = NET_Wrapper(config['WIN_LEN'], config['WIN_OFFSET'])
        network = nn.DataParallel(network)
        network.cuda()

        checkpoint = Checkpoint()
        checkpoint.load(args.model_name)
        network.load_state_dict(checkpoint.state_dict)
        log.info('#' * 14 + 'Finish Resume Model For Test' + '#' * 14)
        print(checkpoint.best_loss)

        # set type and suffix for local test dat
        inpath = config['TEST_PATH']
        lst = [inpath+str(i)+'.wav' for i in range(len(os.listdir(inpath)))]
        test_lst = []
        # test_lst = np.array(test_lst)
        max_len = 0
        for i in lst:
            wav, sr = sf.read(i)
            if len(wav) > max_len:
                max_len = len(wav)
            test_lst.append(torch.FloatTensor(wav))

        dict = {0: 'C', 1: 'E', 2: 'M', 3: 'O'}
        result = ''
        index = 0
        jump = True
        network.eval()
        while jump:
            if index + 10 > len(lst):
                inp = pad_sequence(test_lst[index: len(lst)], batch_first=True)
                jump = False
            inp = pad_sequence(test_lst[index: index+10], batch_first=True)
            index += 10

            # network.eval()
            with torch.no_grad():
                est = network(inp)

            for i in range(est.shape[0]):
                result += dict[est[i].cpu().numpy().argmax()]
        C_n = 0
        for i in range(len(result)):
            if result[i] == 'O':
                C_n += 1
        # requests.post('http://183.175.12.27:9000/push_final/', {'result': result, 'nick': 'ailive', 'sign': 'zhaofei'}).text
        requests.post('http://183.175.12.27:9000/push/', {'result': result, 'nick': 't_resnet_1', 'sign': 'zhaofei'}).text
