import torch
import torch.nn as nn
from decoder_layer import DecoderLayer
from sequential_decoder import SequentialDecoder




class Decoder(nn.Module):
    def __init__(self, input_shape, model_dim, hidden_fc, n_heads, drop_out_prob=0.1, n_layers=1):
        super(Decoder, self).__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(input_shape, model_dim, hidden_fc, n_heads, drop_out_prob)
                                          for _ in range(n_layers)])


    def forward(self, x, y, mask):
        y = self.layers(x, y, mask)
        return y






if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    drop_prob = 0.1
    batch_size = 30
    max_sequence_length = 100
    hidden_fc = 2048
    num_layers = 2
    
    x = torch.randn( (batch_size, max_sequence_length, d_model) ) 
    y = torch.randn( (batch_size, max_sequence_length, d_model) ) 
    mask = torch.full([max_sequence_length, max_sequence_length] , float('-inf'))
    mask = torch.triu(mask, diagonal=1)
    decoder = Decoder(y.shape, d_model, hidden_fc, num_heads, drop_prob, num_layers)
    out = decoder(x, y, mask)
    print(out.shape)