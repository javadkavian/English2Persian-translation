import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def apply_attention(query, key, value, mask=None):
    key_dimmension = query.shape[-1]
    scaled = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(key_dimmension)
    if mask is not None:
        scaled += mask

    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, value)
    return values






class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, model_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.input_dimmension = input_dim
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        self.qkv_generator = nn.Linear(input_dim, 3*model_dim)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, x, mask=None):
        batch_size, sequence_len, _ = x.shape
        qkv = self.qkv_generator(x)
        qkv = qkv.reshape(batch_size, self.n_heads, sequence_len, 3*self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        values = apply_attention(q, k, v, mask)
        values = values.reshape(batch_size, sequence_len, self.n_heads * self.head_dim)    
        out = self.final_linear(values)
        return out




if __name__ == "__main__":
    query = torch.randn((1, 8, 4, 64))
    key = torch.randn((1, 8, 4, 64))
    value = torch.randn((1, 8, 4, 64))
    print("--values befor attention--")
    print(value.shape)
    print("--values after attention--")
    values = apply_attention(query, key, value)
    print(values.shape)
    print("--values after stacking up attention layers--")
    values = values.reshape(1, 4, 8*64)
    print(values.shape)

    input_dim = 1024
    d_model = 512
    num_heads = 8

    batch_size = 30
    sequence_length = 5
    x = torch.randn( (batch_size, sequence_length, input_dim) )

    model = MultiHeadAttention(input_dim, d_model, num_heads)
    out = model.forward(x)

