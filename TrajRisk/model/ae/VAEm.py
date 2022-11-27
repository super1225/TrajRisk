from cProfile import label
import logging
from pickle import LONG
from re import A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
from .base_model import Encoder,Decoder
from model.embedding.gener_embedding_traj import Gener_embedding_traj

class LatentGaussianMixture:
    def __init__(self, args):
        self.args = args
        self.mu_c = torch.Tensor(args.gaussian_cluster_num_m, args.encoder_h_dims_m)
        torch.nn.init.uniform_(self.mu_c,a=0,b=1)
        self.log_sigma_sq_c = torch.Tensor(args.gaussian_cluster_num_m, args.encoder_h_dims_m)
        torch.nn.init.constant_(self.log_sigma_sq_c,0.0)
        self.fc_mu_z = torch.nn.Linear(args.encoder_h_dims_m,args.encoder_h_dims_m)
        initial_fc_mu_z_Wight = torch.Tensor(args.encoder_h_dims_m, args.encoder_h_dims_m).cuda()
        torch.nn.init.normal_(initial_fc_mu_z_Wight,mean=0,std=0.02)
        initial__fc_mu_z_Bias = torch.Tensor(args.encoder_h_dims_m).cuda()
        torch.nn.init.constant_(initial__fc_mu_z_Bias,0.0)
        self.fc_mu_z.weight = torch.nn.Parameter(initial_fc_mu_z_Wight)
        self.fc_mu_z.bias = torch.nn.Parameter(initial__fc_mu_z_Bias)
        self.fc_sigma_z = torch.nn.Linear(args.encoder_h_dims_m,args.encoder_h_dims_m)
        initial_fc_sigma_z_Wight = torch.Tensor(args.encoder_h_dims_m, args.encoder_h_dims_m).cuda()
        torch.nn.init.normal_(initial_fc_sigma_z_Wight,mean=0,std=0.02)
        initial_fc_sigma_z_Bias = torch.Tensor( args.encoder_h_dims_m).cuda()
        torch.nn.init.constant_(initial_fc_sigma_z_Bias,0.0)
        self.fc_sigma_z.weight = torch.nn.Parameter(initial_fc_sigma_z_Wight) 
        self.fc_sigma_z.bias = torch.nn.Parameter(initial_fc_sigma_z_Bias) 

    def post_sample(self, embeded_state, return_loss=False):
        args = self.args
        if(len(embeded_state.shape)==1):
            embeded_state = embeded_state.unsqueeze(0)
        if embeded_state.shape == 3:
            #print(embeded_state.shape[0])
            embeded_state = embeded_state[embeded_state.shape[0]-1]
        mu_z = self.fc_mu_z(embeded_state)
        log_sigma_sq_z = self.fc_sigma_z(embeded_state)
        eps_z = torch.Tensor(log_sigma_sq_z.shape).cuda()
        eps_z = torch.nn.init.normal_(eps_z, mean=0.0, std=1.0)
        z = mu_z + torch.sqrt(torch.exp(log_sigma_sq_z)) * eps_z
        stack_z = torch.stack([z] * args.gaussian_cluster_num_m, axis=1).cuda()
        stack_mu_c = torch.stack([self.mu_c] * z.shape[0], axis=0).cuda()
        stack_mu_z = torch.stack([mu_z] * args.gaussian_cluster_num_m, axis=1).cuda()
        stack_log_sigma_sq_c = torch.stack([self.log_sigma_sq_c] * z.shape[0], axis=0).cuda()
        stack_log_sigma_sq_z = torch.stack([log_sigma_sq_z] * args.gaussian_cluster_num_m, axis=1).cuda()
        pi_post_logits = - torch.sum(torch.square(stack_z - stack_mu_c) / torch.exp(stack_log_sigma_sq_c), dim=-1)
        tosoftmax = nn.Softmax(dim=1)
        pi_post = tosoftmax(pi_post_logits) + 1e-10
        if not return_loss:
            return z
        else:
            batch_gaussian_loss = 0.5 * torch.sum(
                    pi_post * torch.mean(stack_log_sigma_sq_c
                        + torch.exp(stack_log_sigma_sq_z) / torch.exp(stack_log_sigma_sq_c)
                        + torch.square(stack_mu_z - stack_mu_c) / torch.exp(stack_log_sigma_sq_c), dim=-1)
                    , dim=-1) - 0.5 * torch.mean(1 + log_sigma_sq_z, dim=-1)

            batch_uniform_loss = torch.abs(torch.mean(torch.mean(pi_post, dim=0) * torch.log(torch.mean(pi_post, dim=0))))
            return z, [batch_gaussian_loss, batch_uniform_loss],mu_z,log_sigma_sq_z

    def prior_sample(self):
        pass



class VAEmModel(nn.Module):
    def __init__(self,args, config, input_dim, output_dim, embbeding_size, h_dims, gaussian_cluster_num,graph_cluster_num, h_activ=nn.Sigmoid(), out_activ=nn.Tanh()):
        super().__init__()
        self.args = args
        self.vaetype = "move"
        self.embedding = Gener_embedding_traj(config)
        self.encoder = Encoder(input_dim, h_dims, h_activ,
                               out_activ)
        self.decoder = Decoder(config,self.vaetype,self.embedding,input_dim, h_dims, h_activ,out_activ)
        self.classsifier = nn.Linear(h_dims,2)

        self.latent_space = LatentGaussianMixture(args)
        self.out_w = torch.Tensor(args.batch_size, h_dims)
        torch.nn.init.normal_(self.out_w,mean=0,std=0.02)
        self.out_b = torch.Tensor(args.batch_size)
        torch.nn.init.constant_(self.out_b,0.0)
        self.loss_mse = torch.nn.MSELoss(reduction='mean').cuda()
        self.loss_cross = torch.nn.CrossEntropyLoss().cuda()
        
    def forward(self,inputs_normal,label,sequence_len,sub_graph_edges=None):
        
        args = self.args
        sequence_len = inputs_normal["length"]
        #inputs_normal_pad,len = pad_packed_sequence(inputs_normal, batch_first=True)
        inputs = self.embedding(inputs_normal)
        encoder_final_state,out = self.encoder(inputs)
        z, latent_losses,mu_z,log_sigma_sq_z = self.latent_space.post_sample(encoder_final_state[0].squeeze(), return_loss=True)
        outputs = self.decoder(inputs,label,z,encoder_final_state,None)#以实际序列作为输入，生成下个结点
        loss, rec_loss = self.loss(outputs,sequence_len,latent_losses,label)
        return loss, rec_loss



    def loss(self, outputs,sequence_len,latent_losses,label):
        args = self.args
       
        batch_gaussian_loss, batch_uniform_loss = latent_losses
        #mse_loss = 0
        cro_loss = 0
        batch_size = outputs[0].shape[0]
        seq_len = outputs[0].shape[1]
        speed = outputs[0]
        dis = outputs[1]
        label_speed = label[0]
        label_dis =  label[1]
        for i, traj_len in enumerate(sequence_len.tolist()):
            cro_loss = cro_loss + \
                (self.loss_cross(speed[i, :int(traj_len-2)].float(),label_speed[i, :int(traj_len-2)].long()))+(self.loss_cross(dis[i, :int(traj_len-2)].float(),label_dis[i, :int(traj_len-2)].long()))/int(traj_len-2)

        rec_loss = cro_loss/batch_size
        gaussian_loss = torch.mean(batch_gaussian_loss)
        uniform_loss = torch.mean(batch_uniform_loss)
        if args.gaussian_cluster_num_m == 1:
            loss = rec_loss + gaussian_loss
        else:
            print("rec_loss",0.00001*rec_loss)
            print("self.args.encoder_h_dims * gaussian_loss",10.0 / (self.args.encoder_h_dims_m * gaussian_loss))
            print("uniform_loss",uniform_loss)
            print("*"*10,)
           
            loss = 0.00001*rec_loss + 100.0 / self.args.encoder_h_dims_m * gaussian_loss + 10.0 * uniform_loss
        
        return loss, rec_loss

