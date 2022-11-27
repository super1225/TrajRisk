from distutils.command.config import config
import os
import torch
import json
import csv
import pdb
from torch.utils.data import Dataset
import numpy as np
import random
import math
import copy
from torch.nn import functional

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class DatasetTrainRoute(Dataset):  # 继承dataset类

    def __init__(self, config, file_directory, len_threshold=20, train=False):
        print("dataset init")
        self.config = config
        self.routeid_num = config['routeid_num']
        self.len_threshold = len_threshold
        self.file_directory = file_directory
        self.files = os.listdir(file_directory)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):  
        while (True):  
            with open(self.file_directory + "/" + self.files[item]) as f:
                line_org = json.load(f)
            line = []
            if len(line_org) > self.len_threshold and len(line_org) <400 :
                line = copy.deepcopy(line_org)
                break
            else:
                item = (item + 1) % self.__len__()
        label = torch.tensor(line, dtype=torch.int32).unsqueeze(-1)
        line_o = copy.deepcopy(label)/(self.routeid_num+2)
        return [line_o,label]

