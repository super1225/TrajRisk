import os
import math
from re import T
import time
import torch
import argparse
import json
import torch.nn as nn
import numpy as np
from model.ae.VAEr import VAErModel
from model.ae.VAEm import VAEmModel
from model.ae.FusionModel import Fusion_Model
from dataset.dataset_train_traj import DatasetTrainTraj
from dataset.dataset_train_route import DatasetTrainRoute
from dataset.dataset_train_fusion import DatasetTrainFusion
from torch.utils.data import DataLoader
from trainer import Trainer_VAEr,Trainer_VAEm,Trainer_FM
from torch.nn.utils.rnn import pad_sequence
from model.embedding.gener_embedding import Gener_embedding
from memory_profiler import profile

from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = '1'#'0,1'


with open("./config.json") as f:
    config = json.load(f)

adj_file = open(config["adj_path"])
node_adj = json.load(adj_file)

def get_adj(points):
    sample_neighs = []
    for point in points:
        if str(point) in list(node_adj.keys()):
            neighs = node_adj[str(point)]
        else:
            neighs = []
        sample_neighs.append(set(neighs))    
    column_indices = [n for sample_neigh in sample_neighs for n in sample_neigh]
    row_indices = [points[i] for i in range(len(points)) for j in range(len(sample_neighs[i]))]
    sub_graph_edges = torch.LongTensor([row_indices, column_indices])
    return sub_graph_edges

def collate_fn_route(data):
    print("step data load")
    label = []
    data_for_adj = []
    adj = [] 
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_nml = [item[0] for item in data]
    label = [item[1] for item in data]
    seq_len = [len(item[0]) for item in data]
    max_len = max(seq_len)
    for item in label:
        data_for_adj.extend(item.reshape(-1,1).squeeze(-1).tolist()) 
    routid_list = list(set(data_for_adj))
    sub_graph_edges = get_adj(routid_list)
   
    data_nml_pad = pad_sequence(data_nml, batch_first=True, padding_value=(config["routeid_num"])/(config["routeid_num"]+2))
    label = pad_sequence(label, batch_first=True, padding_value=config["routeid_num"])
    return [data_nml_pad,label,sub_graph_edges,seq_len]


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("-em", "--embed_size", type=int, default=64, help="size of subgraph")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=500, help="number of epochs") 
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[1], help="CUDA device ids")
    
    #VAEr parameters setting
    parser.add_argument("-ir", "--encoder_input_dim_r", type=int, default=64, help="input size of lstm model")
    parser.add_argument("-hsr", "--encoder_h_dims_r", type=int, default=64, help="hidden size of lstm model")
    parser.add_argument("-or", "--encoder_output_dim_r", type=int, default=64, help="output size of lstm model")
    parser.add_argument("-cnr", "--gaussian_cluster_num_r", type=int, default=5, help="cluster_num of mixgauss")
    parser.add_argument("-gcn", "--graph_cluster_num", type=int, default=20, help="cluster_num of graph")

    #VAEm parameters setting
    parser.add_argument("-im", "--encoder_input_dim_m", type=int, default=24, help="input size of lstm model")
    parser.add_argument("-hsm", "--encoder_h_dims_m", type=int, default=64, help="hidden size of lstm model")
    parser.add_argument("-om", "--encoder_output_dim_m", type=int, default=64, help="output size of lstm model")
    parser.add_argument("-cnm", "--gaussian_cluster_num_m", type=int, default=5, help="cluster_num of mixgauss")
    
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--train_mode", type=int, default=0, help="0)route sequence train, 1)gps trajectory, 2)test")
    parser.add_argument("--load_file", type=str, default=None)
    parser.add_argument("--output_path_m", type=str, default='save/', help="ex)output/bert.model")
    parser.add_argument("--output_path_r", type=str, default='save/', help="ex)output/bert.model")
    parser.add_argument("--output_path_f", type=str, default='save/', help="ex)output/bert.model")
 
    args = parser.parse_args()
    print("args_train",args)
    
    if args.train_mode == 0:
        print("Loading Train Dataset", config['train_dataset_route'])
        train_dataset = DatasetTrainRoute(config, config['train_dataset_route'], train=True)
        print("Loading Test Dataset", config['test_dataset_route'])
        test_dataset = DatasetTrainRoute(config, config['test_dataset_route'], train=False)
        print("Creating Dataloader")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=collate_fn_route, num_workers=args.num_workers)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size,collate_fn=collate_fn_route, num_workers=args.num_workers) \
            if test_dataset is not None else None
        print("Building model")
        model = VAErModel(args,config, args.encoder_input_dim_r, args.encoder_output_dim_r,args.embed_size, args.encoder_h_dims_r,args.gaussian_cluster_num_r,args.graph_cluster_num)
    writer = SummaryWriter(log_dir="runs")    
    if args.train_mode == 1:
        print("Loading Train Traj Dataset", config['train_dataset_traj'])
        train_dataset = DatasetTrainTraj(config, config['train_dataset_traj'], train=True)
        print("Loading Test Traj Dataset", config['test_dataset_traj'])
        test_dataset = DatasetTrainTraj(config, config['test_dataset_traj'], train=False)
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size,num_workers=args.num_workers) \
            if test_dataset is not None else None
        print("Building model")
        model = VAEmModel(args,config, args.encoder_input_dim_m, args.encoder_output_dim_m,args.embed_size, args.encoder_h_dims_m,args.gaussian_cluster_num_m,args.graph_cluster_num)
    elif args.train_mode == 2:
        print("Loading Train label Dataset", config['train_dataset_label'])
        train_dataset = DatasetTrainFusion(config, config['train_dataset_label'], train=True)
        print("Loading Test label Dataset", config['test_dataset_label'])
        test_dataset = DatasetTrainFusion(config, config['test_dataset_label'], train=False)
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size,num_workers=args.num_workers) \
            if test_dataset is not None else None
        print("Building model")
        model = Fusion_Model(args, config, args.embed_size, args.encoder_input_dim_r, args.encoder_output_dim_r,args.encoder_h_dims_r,args.graph_cluster_num,args.encoder_input_dim_m, args.encoder_output_dim_m,args.encoder_h_dims_m)

    print("Creating Trainer")
    if args.train_mode == 0:
        trainer =Trainer_VAEr(model,hidden_size = args.encoder_h_dims_r,output_size=args.encoder_output_dim_r, train_dataloader=train_data_loader, test_dataloader=test_data_loader,lr=args.lr, weight_decay=args.adam_weight_decay,
                            with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, batch_size = args.batch_size,
                            train_mode = args.train_mode, load_file = args.load_file, output_path = args.output_path_r, config = config)
        print("Training Start")
        for epoch in range(args.epochs):
            trainer.train(writer,epoch)
            trainer.save(epoch, args.output_path_r)
    elif args.train_mode == 1:
        trainer =Trainer_VAEm(model,hidden_size = args.encoder_h_dims_m,output_size=args.encoder_output_dim_m, train_dataloader=train_data_loader, test_dataloader=test_data_loader,lr=args.lr, weight_decay=args.adam_weight_decay,
                            with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, batch_size = args.batch_size,
                            train_mode = args.train_mode, load_file = args.load_file, output_path = args.output_path_m, config = config)
        print("Training Start")
        for epoch in range(args.epochs):
            trainer.train(writer,epoch)
            trainer.save(epoch, args.output_path_m)
    elif args.train_mode == 2:
        trainer =Trainer_FM(model,hidden_size_r = args.encoder_h_dims_r,output_size_r=args.encoder_output_dim_r,hidden_size_m = args.encoder_h_dims_m,output_size_m=args.encoder_output_dim_m, train_dataloader=train_data_loader, test_dataloader=test_data_loader,lr=args.lr, weight_decay=args.adam_weight_decay,
                            with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, batch_size = args.batch_size,
                            train_mode = args.train_mode, load_file = args.load_file, output_path = args.output_path_f, config = config)
        print("Training Start")
        for epoch in range(args.epochs):
            trainer.train(writer,epoch)
            trainer.save(epoch, args.output_path_f)
           
if __name__ == '__main__':
    train()