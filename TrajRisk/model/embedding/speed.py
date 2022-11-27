import torch.nn as nn


class SpeedEmbedding(nn.Embedding):
    def __init__(self, speed_num=30,embed_size=8):
        super().__init__(speed_num, embed_size, padding_idx=0)
