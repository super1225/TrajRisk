import os
import torch
import json
import csv
import pdb
from torch.utils.data import Dataset
import numpy as np
import math
import copy
import sys
from .utils import *

class DatasetTrainTraj(Dataset):

    def __init__(self, config, file_directory, len_threshold=20, train=False):
        
        self.config = config
        self.lines_base = []
        self.lines_static = []
        self.tmp_base = []
        self.tmp_static = []
        self.config=config
        csv.field_size_limit(sys.maxsize)

        self.lines_base = csv.reader(open(file_directory, 'r'))
        self.lines_base = list(self.lines_base)
        for i in range(0,len(self.lines_base)):
            self.lines_base.append(self.lines_base[i])
    
        self.lines_num = len(self.lines_base)
        self.max_length = config['max_len']
    
    def __len__(self):
        return self.lines_num
    
    def __getabnum__(self):
        return self.ab_num
    
    def __getnornum__(self):
        return self.nor_num

    def __getitem__(self, item):
        
        #position feature
        order_base = self.lines_base[item]
        order_base_id, raw_line_base = order_base[0], order_base[1]
        raw_line_base = json.loads(raw_line_base)
        line = line_preprocess_traj(self.config,raw_line_base)
        length = len(line)
        PAD = [self.config["time_num"]+1,self.config["dis_max"]+1,self.config["avg_speed_max"]+1]
        if length > self.max_length:
            line = line[:self.max_length+1]
        else:
            line += [PAD]*(self.max_length+1-len(line))
        line = torch.tensor(line, dtype=torch.float)
        line = line.permute(1, 0)
        output = {
            "time": line[0],
            "dis": line[1],
            "speed": line[2],
            "line": line,
            "id": int(order_base_id),
            "length": length
            }
        
        
        
        return {key: torch.as_tensor(value) for key, value in output.items()}
