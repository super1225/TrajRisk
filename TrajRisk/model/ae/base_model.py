from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import json
import copy
from memory_profiler import profile
import torch.nn.functional as F
from model.embedding.gener_embedding import Gener_embedding
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence, pad_sequence

class Encoder(nn.Module):
    def __init__(self, input_dim, h_dims, h_activ, out_activ):
        super(Encoder, self).__init__()
        self.layer_num = 1
        self.inputdim = input_dim
        self.layer = nn.LSTM(
            input_size=input_dim,
            hidden_size=h_dims,
            num_layers=self.layer_num,
            batch_first=True
        )

        self.h_activ, self.out_activ = h_activ, out_activ
    
    def forward(self, x):
        self.layer.flatten_parameters()
        x, (enc_hidden,cell_data) = self.layer(x)
        out = self.out_activ(enc_hidden.squeeze())
        return (enc_hidden,cell_data),out



class Decoder(nn.Module):
    def __init__(self,config,vaetype,embedding,input_dim, h_dims, h_activ,out_active):
        super(Decoder, self).__init__()
        self.vaetype = vaetype
        self.layer_num =1
        self.config = config
        self.embedding = embedding
        self.embed2latent = nn.Linear(72,64)
        self.layer = nn.LSTM(
            input_size=input_dim,
            hidden_size=h_dims,
            num_layers=self.layer_num,
            batch_first=True
        )
        self.outroute = nn.Linear(h_dims,self.config['routeid_num'])
        self.outspeed = nn.Linear(h_dims,self.config['speed_num'])
        self.outdis = nn.Linear(h_dims,self.config['dis_num'])
        self.out_activ = out_active
        self.pro_net = nn.Softmax(dim=-1)
    def forward(self,input_data, label, enc_hidden,init_state,sub_graph_edges):
        dec_hidden = enc_hidden
        dec_state = (enc_hidden.unsqueeze(0),init_state[1])
        input_orig = copy.deepcopy(label)
        batch_size = input_data.shape[0]
        seq_len = input_data.shape[1]
        if self.vaetype == "route":
            x, (enc_hidden,cell_data) = self.layer(self.embedding(input_orig), dec_state)
            out = self.outroute(x)
            label = label.view(-1)
            source, target = sub_graph_edges
            # (batch_size*seq_len, sub_edge_num)
            source = source.unsqueeze(0).repeat(label.shape[0], 1)
            target = target.unsqueeze(0).repeat(label.shape[0], 1)
            # (batch_size*seq_len, sub_edge_num)
            source = (source!=(label.unsqueeze(1).repeat(1, sub_graph_edges.shape[1]))).long()
            # (batch_size*seq_len, label_num)
            mask = torch.zeros(label.shape[0], self.config['routeid_num']).long().to(source.device)
            # mask[i][target[i][j]] = src[i][j]
            mask.scatter_add_(dim=1, index=torch.tensor(target, dtype=torch.int64), src=source)
            # (batch_size*seq_len, node_num) => (batch_size, seq_len, node_num)
            mask = mask.view(batch_size, seq_len, -1).contiguous()
            mask = -9999999*mask
            out_mask = mask + out
            return out_mask
        elif self.vaetype == "move":
            for i in range(0,seq_len-1):
                pred_step = []
                x, (enc_hidden,cell_data) = self.layer(input_data, dec_state)
                speed = self.outspeed(x)
                dis = self.outdis(x)
                
            return [speed, dis]
        
    
