from sys import _xoptions
from tkinter import Y
from urllib import robotparser
import torch.nn as nn
#import keras.backend as k
import torch
import numpy as np
import copy
import pdb
from operator import itemgetter
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence
from .timestamp import TimeEmbedding
from .roadgcn import Road_Embedding



class Gener_embedding(nn.Module):

    def __init__(self, config, embed_size,cluster_num, dropout=0.1):
        super().__init__()
        self.config = config
        self.road_embedding = Road_Embedding(config,embed_size,cluster_num)
        self.embed_size = embed_size
        self.linear_routeid = nn.Linear(1,16)

    def forward(self, data_orig):
        road_batch_embedding = self.get_road_embedding(data_orig)
        return road_batch_embedding
        return self.linear_routeid(data_normal)
        road_batch_embedding = self.get_road_embedding(data_orig)
        x_pro = torch.cat([self.linear_routeid(data_normal), road_batch_embedding], dim=-1)
        return x_pro
        
    def get_road_embedding(self, data):
        road_map = self.road_embedding.road_map
        sub_embedding = {}
        roudid_batch = data.reshape(-1).tolist()
        roudid_batch_t = []#
        roudid_batch_int = []
        flag = 0
        for i in roudid_batch:
            if i != self.config['routeid_num']+1:
                roudid_batch_int.append(int(i))
        if len(roudid_batch_int) != 0:#
            if len(roudid_batch_int) == 1:
                clusterid_inbatch = [road_map[roudid_batch_int[0]]]
            else:
                clusterid_inbatch = list(set(itemgetter(*roudid_batch_int)(road_map)))#
            sub_embedding = self.road_embedding(clusterid_inbatch)
     
        for routeid in roudid_batch:#
            if routeid == self.config['routeid_num']+1:
                roudid_batch_t.append(torch.Tensor(self.embed_size).zero_().cuda())
            else:
                roudid_batch_t.append(sub_embedding[road_map[routeid]])
        if len(data.shape)==3: 
            roudid_batch_t = torch.stack(roudid_batch_t,0).reshape(data.shape[0],data.shape[1],-1)
        else:
            roudid_batch_t = torch.stack(roudid_batch_t,0).reshape(data.shape[0],data.shape[1],data.shape[2],-1)
        return roudid_batch_t
        return x

if __name__ == '__main__':
    embedding = Gener_embedding(200 * 120, 4, 12, 512)