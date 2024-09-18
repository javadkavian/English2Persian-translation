import torch
import torch.nn as nn
from encoder_layer import EncoderLayer




class Encoder(nn.Module):
    
    def __init__(self, input_shape, model_dim, hidden_fc, n_heads ,drop_out_prob, n_layers):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.Sequential(*[EncoderLayer(input_shape, model_dim, hidden_fc, n_heads, drop_out_prob)
                                            for _ in range(n_layers)])
        

    def forward(self, x):
        x = self.encoder_layers(x)
        return x





if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    drop_prob = 0.1
    batch_size = 30
    max_sequence_length = 200
    ffn_hidden = 2048
    num_layers = 5
    x = torch.randn( (batch_size, max_sequence_length, d_model) ) 
    encoder = Encoder(x.shape, d_model, ffn_hidden, num_heads, drop_prob, num_layers)
    out = encoder(x)
    print(out.shape)