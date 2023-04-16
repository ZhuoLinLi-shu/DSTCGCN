import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class FFTSelector(nn.Module):
    def __init__(self,feature_num,hidden_map_dim):
        super(FFTSelector, self).__init__()
        self.hidden_map_q = nn.Linear(feature_num,hidden_map_dim)
        self.hidden_map_k = nn.Linear(feature_num,hidden_map_dim)
        self.acti = nn.Softmax(dim=-1)

    def forward(self, cat_x, tau):
        # cat_x: B,T,N,D (D=2 for pems04)
        # tau: number of relative time steps
        # ps: multi-heads will be supported
        time_len = cat_x.shape[1]
        features_q = self.hidden_map_q(cat_x)
        features_k = self.hidden_map_k(cat_x)

        score_set = torch.zeros(time_len,time_len) # T,T

        k_fft = torch.fft.rfft(features_k)

        for i in range(time_len):
            features_q_i = features_q[:,i:i+1,:,:] # B,1,N,d_f
            features_q_i = features_q_i.repeat(1,time_len,1,1) # B,T,N,d_f
            q_fft_i = torch.fft.rfft(features_q_i)
            res = q_fft_i * torch.conj(k_fft)
            corr = torch.fft.irfft(res, dim=-1)
            mean_value = torch.mean(torch.mean(torch.mean(corr, dim=0), dim=-1),dim=-1)
            score_set[i] = mean_value
        scores, indices = torch.topk(score_set,tau)
        
        return scores, indices



class TNorm(nn.Module):
    def __init__(self,  num_nodes, channels, track_running_stats=True, momentum=0.1):
        super(TNorm, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer('running_var', torch.ones(1, channels, num_nodes, 1))
        self.momentum = momentum

    def forward(self, x):
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)

        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out

