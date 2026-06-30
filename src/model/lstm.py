import torch
import torch.nn as nn
import torch.nn.functional as F
import math





class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.lstm = torch.nn.LSTM(
            input_size = configs.enc_in,
            hidden_size = configs.d_model, 
            num_layers = configs.num_layers,
            dropout = configs.dropout,
            batch_first = True,
        )
        
        self.projection = nn.Sequential(
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, configs.enc_in)
        )
        
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.configs.task_name == 'long_term_forecast' or self.configs.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B,pred_len,C]
            return dec_out
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        o, _ = self.lstm(x_enc)
        dec_out = self.projection(o[:,-1:])
        
        # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        return dec_out
