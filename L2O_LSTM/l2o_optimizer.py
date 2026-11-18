#!/usr/bin/python3
# --------------------------------------------------------- #
#  LSTM Optimizer Model                                     #
# --------------------------------------------------------- #

import torch
import torch.nn as nn

class L2O_BlackBox_LSTM(nn.Module):
    def __init__(self, hidden_sz=20, num_layers=2):
        super().__init__()
        self.hidden_sz = hidden_sz
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_sz, num_layers=num_layers, batch_first=False)
        self.output = nn.Linear(hidden_sz, 1)

    def forward(self, seq_dx, seq_df):
        seq_len, n = seq_dx.shape
        
        # Normalization of the input sequences to avoid exploding gradients
        in_scale_dx = torch.max(torch.abs(seq_dx), dim=0, keepdim=True)[0] + 1e-8
        in_scale_df = torch.max(torch.abs(seq_df)) + 1e-8
        
        norm_dx = seq_dx / in_scale_dx
        norm_df = seq_df / in_scale_df
        
        dx_reshaped = norm_dx.unsqueeze(-1)
        df_reshaped = norm_df.view(seq_len, 1, 1).expand(seq_len, n, 1)
        
        lstm_in = torch.cat([dx_reshaped, df_reshaped], dim=-1)
        
        out, _ = self.lstm(lstm_in)
        last_out = out[-1] 
        
        norm_step = self.output(last_out) 
        
        out_scale_dx = torch.mean(torch.abs(seq_dx[-2:]), dim=0, keepdim=True) + 1e-8
        step_pred = norm_step * out_scale_dx.T
        
        return step_pred