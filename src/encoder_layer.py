import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from layer_normalization import LayerNormalization



class EncoderLayer(nn.Module):
    def __init__(self, input_shape, model_dim, hidden_fc, n_heads = 8, drop_out_porb = 0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim=model_dim, n_heads=n_heads)
        self.norm1 = LayerNormalization(input_shape=input_shape)
        self.dropout1 = nn.Dropout(p=drop_out_porb)





