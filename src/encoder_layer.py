import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from layer_normalization import LayerNormalization
from encoder_feed_forward import EncoderFeedForward



class EncoderLayer(nn.Module):#input_dim and model dim should be equal.test the case where they are not with nn.Linear
    #also model dim should be coefficient of number of heads
    def __init__(self, input_shape, model_dim, hidden_fc, n_heads = 8, drop_out_prob = 0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(input_dim= input_shape[-1], model_dim=model_dim, n_heads=n_heads)
        self.norm1 = LayerNormalization(input_shape=input_shape)
        self.dropout1 = nn.Dropout(p=drop_out_prob)
        self.fc = EncoderFeedForward(model_dim, hidden_fc, drop_out_prob)
        self.norm2 = LayerNormalization(input_shape)
        self.dropout2 = nn.Dropout(p=drop_out_prob)

    def forward(self, x):
        residual_path = x
        x = self.attention(x)
        x = self.dropout1(x)
        x = self.norm1(x + residual_path)
        residual_path = x 
        x = self.fc(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_path)
        return x





if __name__ == "__main__":
    batch_size = 3
    sequence_len = 5
    embedding_dim = 64
    model_dim = 64
    hidden_fc = 200
    x = torch.randn((batch_size, sequence_len, embedding_dim))
    model = EncoderLayer(x.shape, model_dim, hidden_fc)

    print(model(x).shape)




