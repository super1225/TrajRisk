import torch.nn as nn


class TimeEmbedding(nn.Embedding):
    def __init__(self, time_num, embed_size=8):
        super().__init__(time_num, embed_size, padding_idx=0)
