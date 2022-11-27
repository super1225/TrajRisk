import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import trange, tqdm
from .layers import StackedGCN
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class ClusterGCN(nn.Module):
    """
    Training a ClusterGCN.
    """
    def __init__(self, clustering_machine,embed_size):
        """
        :param ags: Arguments object.
        :param clustering_machine:
        """ 
        super().__init__() 
        self.clustering_machine = clustering_machine
        self.model = StackedGCN(self.clustering_machine.feature_count, embed_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, cluster):
        self.model = self.model.to(self.device)
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        train_nodes = self.clustering_machine.sg_train_nodes[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
       
        sub_embedding = self.model(edges, features)
      
        return sub_embedding

   