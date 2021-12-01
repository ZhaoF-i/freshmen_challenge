import torch.nn as nn
import torch.autograd.variable
import numpy as np
from torch.autograd.variable import *

from utils.stft_istft import STFT


class NET_Wrapper(nn.Module):
    def __init__(self,win_len,win_offset):
        super(NET_Wrapper, self).__init__()

        self.win_len = win_len
        self.win_offset = win_offset
        self.STFT = STFT(self.win_len, self.win_offset).cuda()
        self.d_model = 256
        self.d_f = 64
        self.k = 3
        self.max_d_rate = 16
        self.layer_list = []
        self.dila_list = nn.ModuleList([])
        for i in range(5):
            self.dila_list.append(nn.Conv1d(in_channels=self.d_f, out_channels=self.d_f, kernel_size=self.k
                       , padding=int(2 ** i), dilation=int(2 ** i)))
        self.conv1_outp = nn.Conv1d(in_channels=self.d_model, out_channels=161, kernel_size=1)
        self.conv1_inp = nn.Conv1d(in_channels=161, out_channels=self.d_model, kernel_size=1)
        self.conv1_dm = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_f, kernel_size=1)
        self.conv1_df = nn.Conv1d(in_channels=self.d_f, out_channels=self.d_model, kernel_size=1)

        self.BN_dm = nn.BatchNorm1d(self.d_model)
        self.BN_df = nn.BatchNorm1d(self.d_f)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, input_data_c1):
        # input_data_c1 = train_info_.mix_feat_b
        STFT_C1, phase_C1 = self.STFT.transform(input_data_c1)
        # STFT_C1 = STFT_C1.permute(0, 2, 1)
        input_feature = STFT_C1



        # self.layer_list.append(self.relu(self.BN_dm(self.conv1_inp(input_feature))))
        conv = self.conv1_inp(input_feature)
        conv = self.BN_dm(conv)
        # conv = nn.LayerNorm(normalized_shape=conv.shape[-1], eps=1e-6)(conv)
        conv = self.relu(conv)
        self.layer_list.append(conv)

        # feedforward = self.relu(self.BN_dm(self.conv1_inp(input_feature)))
        for i in range(20):
            #self.layer_list.append(self.block(self.layer_list[-1], int(2**(i%(np.log2(self.max_d_rate)+1)))))
            residual = self.layer_list[-1]
            # conv1 = self.conv1_dm(self.relu(self.BN_dm(self.layer_list[-1])))
            conv1 = self.BN_dm(self.layer_list[-1])
            conv1 = self.relu(conv1)
            conv1 = self.conv1_dm(conv1)

            conv2 = self.BN_df(conv1)
            conv2 = self.relu(conv2)
            conv2 = self.dila_list[i % 5](conv2)

            # conv3 = self.conv1_df(self.relu(self.BN_df(conv2)))
            conv3 = self.BN_df(conv2)
            conv3 = self.relu(conv3)
            conv3 = self.conv1_df(conv3)

            self.layer_list.append(residual + conv3)

        outp = self.conv1_outp(self.layer_list[-1])
        outp = outp.permute(0, 2, 1)

        return [outp, STFT_C1, phase_C1]
        # outp = self.sigmoid(outp)


