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

class DatasetTrainFusion(Dataset):

    def __init__(self, config, file_directory, len_threshold=20, train=False):
        
        self.config = config
        self.data_list = []
        self.lines_m = []
        self.lines_static = []
        self.tmp_base = []
        self.tmp_static = []
        self.config=config
        self.len_threshold = len_threshold
        csv.field_size_limit(sys.maxsize)
        self.file_directory_m = file_directory+"move.csv"
        self.file_directory_r = file_directory+"route"
        self.lines_m = csv.reader(open(self.file_directory_m, 'r'))
        self.lines_m = list(self.lines_m)
        for i in range(0,len(self.lines_m)):
            order_id = self.lines_m[i][0]
            raw_data_m = json.loads(self.lines_m[i][1])
            line = line_preprocess_traj(self.config,raw_data_m)
            if os.path.exists(self.file_directory_r + "/" + order_id +".json"):
                with open(self.file_directory_r + "/" + order_id +".json") as f:
                    raw_data_r = json.load(f)
                line = []
                if len(raw_data_r) > self.len_threshold and len(raw_data_r) <400 :
                    line = copy.deepcopy(raw_data_r)
                    continue
                self.data_list.append([raw_data_r,raw_data_m,order_id[-1]])#route,move,label

        self.lines_num = len(self.data_list)
        self.max_length = config['max_len']
    
    def __len__(self):
        return self.lines_num
    
    def __getabnum__(self):
        return self.ab_num
    
    def __getnornum__(self):
        return self.nor_num

    def __getitem__(self, item):
        data = self.data_list[item]
        data_r = data[0]
        data_m = data[1]
        label = int(data[2])
        #route feature
        length_r = len(data_r)
        PAD = [self.config["routeid_num"]+1]
        if  length_r > self.max_length:
            data_r = data_r[:self.max_length+1]
        else:
            data_r += PAD*(self.max_length+1-len(data_r))
        data_r = torch.tensor(data_r, dtype=torch.int32)
        #move feature
        line = line_preprocess_traj(self.config,data_m)
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
            "length": length
            }
        
        
        
        return [[data_r,length_r],{key: torch.as_tensor(value) for key, value in output.items()},label]
