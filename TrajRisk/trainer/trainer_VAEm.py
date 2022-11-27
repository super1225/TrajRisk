from pyexpat import model
from shutil import copy
from typing import Sequence
import torch
import torch.nn as nn
from torch.nn.modules.module import T
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import pdb
import numpy as np
import copy
import json
import math
import os
from model.ae.VAEm import VAEmModel
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence


class Trainer_VAEm:

    def __init__(self, model:VAEmModel,hidden_size,output_size, train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-3, betas=(0.9, 0.999), weight_decay: float = 0.01, with_cuda: bool = True, cuda_devices=None, batch_size: int = 30,
                 train_mode: int = 0, load_file: str = None, output_path: str = None, config: dict = None):

        self.model = model
        self.load_file = load_file

        if load_file != None and load_file[:5] == 'train':  
            self.model.load_state_dict(torch.load(output_path + load_file))

        # Distributed GPU training if CUDA can detect more than 1 GPU
        print(cuda_devices)
        if with_cuda and torch.cuda.device_count() > 1:  
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=[1])
            
        
        self.model = self.model.cuda()
        self.loss_crosse = torch.nn.CrossEntropyLoss().cuda()
        self.loss_mse = torch.nn.MSELoss(reduction='mean').cuda()

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.train_mode = train_mode
        self.config = config
        self.batch_size = batch_size

        print("batchsize:",batch_size)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def dataset_renew(self, train_dataloader, test_dataloader):
        self.train_data = train_dataloader
        self.test_data = test_dataloader

    def train(self,writer, epoch):
        self.iteration(writer,epoch, self.train_data)

    def test(self,writer, epoch):
        self.iteration(writer,epoch, self.test_normal_data, train=False, normal_flag=True)
        self.iteration(writer,epoch, self.test_ab_data, train=False, normal_flag=False)

    def iteration(self,writer, epoch, data_loader, train=True, normal_flag=True):
        str_code = "train" if train else "test"
        # Setting the tqdm progress bar  
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        rec_loss_list= []
        loss_list = []
        for i, data_batch in data_iter:
            data = {key: value.cuda() for key, value in data_batch.items()}
            seq_len = data["id"].size()[0]
            label = [data["speed"],data["dis"]]
            loss_vae,rec_loss =  self.model.forward(data,label,seq_len)   
            loss_list.append(loss_vae.item())
            rec_loss_list.append(rec_loss)
            if train:
                self.optim.zero_grad()  
                loss_vae.backward()  
                self.optim.step()  

        if train:
            writer.add_scalar('train_loss',np.mean(loss_list),epoch)
            writer.add_scalar('train_loss',np.mean(rec_loss_list),epoch)
        else:
            if normal_flag:
                writer.add_scalar('test_normal_loss',np.mean(loss_list),epoch)
            else:
                writer.add_scalar('test_ab_loss',np.mean(loss_list),epoch)
        
            


       
    def save(self, epoch, file_path="output/bert_trained.model"):

        if self.train_mode != 0 and self.train_mode != 1:
            output_path = file_path + "train.task%d.ep%d" % (self.train_mode, epoch)
            torch.save(self.model.module.state_dict(), output_path)
            print("EP:%d Model Saved on:" % epoch, output_path)

            output_path = file_path + "encoder.task%d.ep%d" % (self.train_mode, epoch)
            torch.save(self.model.encoder.state_dict(), output_path)
            print("EP:%d Encoder Saved on:" % epoch, output_path)
            return output_path

        elif self.train_mode == 0 or self.train_mode == 1:
            output_path = file_path + "train.task%d.ep%d" % (self.train_mode, epoch)
            torch.save(self.model.state_dict(), output_path)
            print("EP:%d Model Saved on:" % epoch, output_path)
            return output_path
