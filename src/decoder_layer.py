import torch
import torch.nn  as nn
from multi_head_attention import MultiHeadAttention
from multi_head_cross_attention import MultiHeadCrossAttention
from layer_normalization import LayerNormalization
from encoder_feed_forward import EncoderFeedForward #rename it later




class DecoderLayer(nn.Module):
    def __init__(self, input_shape, model_dim, hidden_fc, n_heads, drop_out_prob=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(input_dim=input_shape[-1], model_dim=model_dim, n_heads=n_heads)
        self.norm1 = LayerNormalization(input_shape)
        self.dropout1 = nn.Dropout(p=drop_out_prob)
        self.cross_attention = MultiHeadCrossAttention(model_dim, n_heads)
        self.norm2 = LayerNormalization(input_shape)
        self.dropout2 = nn.Dropout(p=drop_out_prob)
        self.fc = EncoderFeedForward(model_dim, hidden_fc, drop_out_prob)
        self.norm3 = LayerNormalization(input_shape)
        self.dropout3 = nn.Dropout(p=drop_out_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        # print("--testing inside the model--")
        # print(f'shape of y : {y.shape} and shape of x : {x.shape}')
        residual_path = y.clone()
        y = self.self_attention(y, mask=self_attention_mask)
        # print(f'y after attention : {y.shape}')
        y = self.dropout1(y)
        # print(f'y after dropout : {y.shape}')
        y = self.norm1(y + residual_path)
        # print(f'y after add and norm : {y.shape}')
        residual_path = y.clone()
        y = self.cross_attention(x, y, mask=cross_attention_mask)
        # print(f'y after cross attention : {y.shape}')
        y = self.dropout2(y)
        # print(f'y after dropout : {y.shape}')
        y = self.norm2(y + residual_path)
        # print(f'y after add and norm : {y.shape}')
        residual_path = y.clone()
        y = self.fc(y)
        # print(f'y after fully connected layer : {y.shape}')
        y = self.dropout3(y)
        # print(f'y after dropout : {y.shape}')
        y = self.norm3(y + residual_path)
        # print(f'y after add and norm : {y.shape}')
        return y 




if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    drop_prob = 0.1
    batch_size = 30
    max_sequence_length = 200
    hidden_fc = 50

    x = torch.randn( (batch_size, max_sequence_length, d_model) ) 
    y = torch.randn( (batch_size, max_sequence_length, d_model) )


    decoder = DecoderLayer(y.shape, d_model, hidden_fc, num_heads, drop_prob)
    out = decoder(x, y, None, None)
    print(out.shape)