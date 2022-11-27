import torch.nn as nn


class DisEmbedding(nn.Embedding):
    def __init__(self, dis_num=30, embed_size=8):
        super().__init__(dis_num, embed_size, padding_idx=0)
