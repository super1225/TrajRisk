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
from .speed import SpeedEmbedding
from .dis import DisEmbedding



class Gener_embedding_traj(nn.Module):

    def __init__(self, config, dropout=0.1):
        super().__init__()
        self.config = config
        self.speed = SpeedEmbedding(speed_num=config['speed_num']+2, embed_size=8)
        self.dis = DisEmbedding(dis_num=config['dis_num']+2, embed_size=8)
        self.time = TimeEmbedding(time_num=config['time_num']+2, embed_size=8)

    def forward(self, data):
        x = torch.cat((self.time(data['time'].int()),self.dis((data["dis"]).int()),self.speed((data["speed"]).int())),2)
        return x
        

if __name__ == '__main__':
    embedding = Gener_embedding(200 * 120, 4, 12, 512)