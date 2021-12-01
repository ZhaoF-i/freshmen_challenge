import scipy.io as sio
import soundfile as sf
import numpy as np
import struct
import mmap
import os
from utils.util import dirc_map

from torch.nn.utils.rnn import *
from torch.autograd.variable import *
from torch.utils.data import Dataset, DataLoader


class SpeechMixDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.lst_path = config['TRAIN_LST'] if mode == 'train' else config['CV_LST']
        self.lst = [str(i) for i in range(len(os.listdir(self.lst_path)))]
        # self.lst.sort()
        self.label_txt = config['TRAIN_LABEL_PATH'] if mode == 'train' else config['CV_LABEL_PATH']
        with open(self.label_txt, 'r') as file_to_read:
            self.label = file_to_read.readline()

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        wav, _ = sf.read(self.lst_path + '/' + str(self.lst[idx]) + '.wav')
        label = self.label[idx]
        label = dirc_map(label)

        sample = (Variable(torch.FloatTensor(wav.astype('float32'))),
                  label
                  # Variable(label)
                  )

        return sample


class BatchDataLoader(object):
    def __init__(self, s_mix_dataset, batch_size, is_shuffle=True, workers_num=16):
        self.dataloader = DataLoader(s_mix_dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=workers_num,
                                     collate_fn=self.collate_fn)

    def get_dataloader(self):
        return self.dataloader

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: x[0].size()[0], reverse=True)
        wav, label = zip(*batch)
        wav = pad_sequence(wav, batch_first=True)
        # label = pad_sequence(label, batch_first=True)
        label = torch.LongTensor(label)

        return [wav, label]
