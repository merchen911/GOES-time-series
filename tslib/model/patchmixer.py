# __all__ = ['PatchMixer']

# Cell
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from .layers.PatchTST_layers import *
from .layers.RevIN import RevIN



class PatchMixerLayer(nn.Module):
    def __init__(self,dim,a,kernel_size = 8):
        super().__init__()
        self.Resnet =  nn.Sequential(
            nn.Conv1d(dim,dim,kernel_size=kernel_size,groups=dim,padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim,a,kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(a)
        )
    def forward(self,x):
        x = x +self.Resnet(x)                  # x: [batch * n_val, patch_num, d_model]
        x = self.Conv_1x1(x)                   # x: [batch * n_val, a, d_model]
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = Backbone(configs)
    def forward(self, x, *args):
        x = self.model(x)
        return x


class Backbone(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name

        self.nvals = configs.enc_in
        self.lookback = configs.seq_len
        self.forecasting = configs.pred_len
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.kernel_size = configs.mixer_kernel_size

        self.PatchMixer_blocks = nn.ModuleList([])
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num = int((self.lookback - self.patch_size)/self.stride + 1) + 1
        # if configs.a < 1 or configs.a > self.patch_num:
        #     configs.a = self.patch_num
        self.a = self.patch_num
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.head_dropout = configs.head_dropout
        self.depth = configs.e_layers
        for _ in range(self.depth):
            self.PatchMixer_blocks.append(PatchMixerLayer(dim=self.patch_num, a=self.a, kernel_size=self.kernel_size))
        self.W_P = nn.Linear(self.patch_size, self.d_model)  
        self.head0 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.d_model, self.forecasting),
            nn.Dropout(self.head_dropout)
        )
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.a * self.d_model, int(self.forecasting * 2)),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(int(self.forecasting * 2), self.forecasting),
            nn.Dropout(self.head_dropout)
        )
        self.dropout = nn.Dropout(self.dropout)
        # RevIn
        self.revin = configs.revin
        if self.revin: self.revin_layer = RevIN(self.nvals, affine=configs.affine, subtract_last=configs.subtract_last)
            
        
        if self.task_name == 'regression':
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.forecasting * configs.enc_in, configs.num_class)



    def forecast(self, x):
        bs = x.shape[0]
        nvars = x.shape[-1]
        if self.revin:
            x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1)                                                       # x: [batch, n_val, seq_len]

        x_lookback = self.padding_patch_layer(x)
        x = x_lookback.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # x: [batch, n_val, patch_num, patch_size]  

        x = self.W_P(x)                                                              # x: [batch, n_val, patch_num, d_model]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))      # x: [batch * n_val, patch_num, d_model]
        x = self.dropout(x)
        u = self.head0(x)

        for PatchMixer_block in self.PatchMixer_blocks:
            x = PatchMixer_block(x)
        x = self.head1(x)
        x = u + x
        x = torch.reshape(x, (bs , nvars, -1))                                       # x: [batch, n_val, pred_len]
        x = x.permute(0, 2, 1)
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x





    def regression(self, x):
        bs = x.shape[0]
        nvars = x.shape[-1]
        x = x.permute(0, 2, 1)                                                       # x: [batch, n_val, seq_len]

        x_lookback = self.padding_patch_layer(x)
        x = x_lookback.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # x: [batch, n_val, patch_num, patch_size]  

        x = self.W_P(x)                                                              # x: [batch, n_val, patch_num, d_model]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))      # x: [batch * n_val, patch_num, d_model]
        x = self.dropout(x)
        u = self.head0(x)

        for PatchMixer_block in self.PatchMixer_blocks:
            x = PatchMixer_block(x)
        x = self.head1(x)
        x = u + x
        # x = torch.reshape(x, (bs , nvars, -1))                                       # x: [batch, n_val, pred_len]
        
        # Decoder
        output = x.reshape(bs, -1)
        output = self.dropout(output)
        output = self.projection(output)  # (batch_size, num_classes)
        return output





    def forward(self, x_enc):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.forecasting:, :]  # [B, L, D]
        if self.task_name == 'regression':
            dec_out = self.regression(x_enc)
            return dec_out  # [B, N]
        return None