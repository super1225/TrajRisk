from distutils.command.config import config
import torch.nn as nn
import torch
from .gcn.clustering import ClusteringMachine
from .gcn.clustergcn import ClusterGCN
from .gcn.utils import tab_printer, graph_reader, feature_reader, target_reader


class Road_Embedding(nn.Module):

    def __init__(self,config,embed_size, cluster_num, dropout=0.1):
        super().__init__()
        self.config = config
        print(config['edge_path'])
        self.graph = graph_reader(self.config['edge_path'])
        self.features = feature_reader(self.config['feature_path'])
        self.clustering_machine = ClusteringMachine(cluster_num, self.graph, self.features)
        self.clustering_machine.decompose()
        self.road_map = self.clustering_machine.route_map()
        self.gcn = ClusterGCN(self.clustering_machine,embed_size)
        #self.embedding = Gener_embedding(, embed_size)

    def forward(self,clusterid_inbatch):
        sub_embedding = {}
        for clusterid in clusterid_inbatch:
            sub_embedding[clusterid] = self.gcn(clusterid)
        return sub_embedding

if __name__ == '__main__':
    roadembedding = Road_Embedding()
    roadembedding.forward()
