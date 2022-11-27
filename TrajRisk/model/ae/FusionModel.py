import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from .base_model import Encoder,Decoder
from model.embedding.gener_embedding import Gener_embedding
from model.embedding.gener_embedding_traj import Gener_embedding_traj

class Fusion_Model(nn.Module):
    def __init__(self,args, config,embed_size,input_dim_r, output_dim_r, h_dims_r,graph_cluster_num,input_dim_m, output_dim_m, h_dims_m, h_activ=nn.Sigmoid(),out_active=nn.ReLU()):  # （8，64，6，5，[64,128]）
        super().__init__()
        self.embedding_r = Gener_embedding(config, embed_size,graph_cluster_num)
        self.embedding_m = Gener_embedding_traj(config)
        self.encoder_r = Encoder(input_dim_r, h_dims_r, h_activ,out_active)
        self.encoder_m = Encoder(input_dim_m, h_dims_m, h_activ,out_active)
        self.classsifier = nn.Linear(128,2)
        self.loss_cross = torch.nn.CrossEntropyLoss().cuda()
        
    def forward(self, data_r,sequence_len_r,data_m,label):
        data_r_embedding = self.embedding_r(data_r.unsqueeze(-1))
        enc_hidden,output_r = self.encoder_r(data_r_embedding)  
        data_m_embedding = self.embedding_m(data_m)
        enc_hidden,output_m = self.encoder_m(data_m_embedding)  
        pro = self.classsifier(torch.cat((output_r,output_m),-1))
        loss = self.loss_cross(pro,label)
        return loss

    def pair_cosine_similarity(x, x_adv, eps=1e-8):
        n = torch.norm(x,p=2,dim=1, keepdim=True)
        n_adv = torch.norm(x_adv,p=2,dim=1, keepdim=True)
        return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_adv @ x_adv.t()) / (n_adv * n_adv.t()).clamp(min=eps), (x @ x_adv.t()) / (n * n_adv.t()).clamp(min=eps)