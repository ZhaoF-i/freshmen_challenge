from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import *
import torch
from glob import glob
from tqdm import tqdm
from random import shuffle, sample, randint, choices
from speechbrain.dataio.dataio import read_audio, write_audio
from speechbrain.lobes.augment import TimeDomainSpecAugment
from speechbrain.lobes.augment import EnvCorrupt
from functools import reduce
import torchaudio
import numpy as np


class FPCDataset(Dataset):
    def __init__(self, mode='train'):
        super().__init__()

        # 数据集音频和标签地址
        tra_path = '/data01/fanhaipeng/NewStudent/course-v3/zh-nbs/my_dev/'
        val_path = '/data01/fanhaipeng/NewStudent/course-v3/zh-nbs/my_dev_val/'
        lab_tra_path = '/data01/fanhaipeng/NewStudent/course-v3/zh-nbs/my_lab.txt'
        lab_val_path = '/data01/fanhaipeng/NewStudent/course-v3/zh-nbs/my_lab_val.txt'

        # 训练集or测试集
        if mode == 'validate':
            tra_path = val_path
            lab_tra_path = lab_val_path

        # 初始化音频、标签和字典
        self.src, self.trg = [], []
        tag_dictionary = {
            'M': 0,
            'E': 1,
            'C': 2
        }

        # 读取音频和标签
        dt_data = []
        for i in range(len(glob(tra_path + '*.wav'))):
            dt_data.append(tra_path + '%d.wav' % i)
        with open(lab_tra_path, "r") as f:  # 设置文件对象
            label_txt = f.read()  # 可以是随便对文件的操作
        pack = list(zip(dt_data, label_txt))
        shuffle(pack)
        print('Read audio and labels:')
        for p in tqdm(pack):
            w, lab = p
            wav_data = read_audio(w)
            self.src.append(wav_data)
            self.trg.append(tag_dictionary[lab])

    def __getitem__(self, index):
        return self.src[index], self.trg[index]

    def __len__(self):
        return len(self.src)


class FPCDataLoader(object):
    def __init__(self, s_mix_dataset, batch_size, is_shuffle=True, workers_num=16):
        self.dataloader = DataLoader(s_mix_dataset, batch_size=batch_size, shuffle=is_shuffle,
                                     num_workers=workers_num, collate_fn=self.collate_fn)

    def get_dataloader(self):
        return self.dataloader

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: x[0].size()[0], reverse=True)
        wav, tag = zip(*batch)

        # 左侧填充0使batch长度相同
        wav_size = []
        for i in wav:
            wav_size.append(i.shape[0])
        max_wav_size = max(wav_size)
        wav_pad = []
        for i, i_data in enumerate(wav):
            zero_num = max_wav_size - wav_size[i]
            if zero_num > 0:
                zero = torch.zeros(zero_num)
                wav_pad.append(torch.cat((zero, i_data), dim=0))
            else:
                wav_pad.append(i_data)

        wav_batch = pad_sequence(wav_pad, batch_first=True)

        tag_batch = np.array(tag).astype(int)
        tag_batch = torch.from_numpy(tag_batch)

        # print(tag)
        return [wav_batch, tag_batch]


if __name__ == '__main__':
    data_train = FPCDataset()
    tr_batch_dataloader = FPCDataLoader(data_train, 32, is_shuffle=True, workers_num=8)
    for i_batch, batch_data in enumerate(tr_batch_dataloader.get_dataloader()):
        print(i_batch)  # 打印batch编号
        # print(batch_data[0])  # 打印该batch里面src （会死掉的）
        print(batch_data[0].shape)
        print(batch_data[1])  # 打印该batch里面trg
