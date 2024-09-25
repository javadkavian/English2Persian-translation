import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def apply_attention(query, key, value, mask=None):
    key_dimmension = query.shape[-1]
    scaled = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(key_dimmension)
    if mask is not None:
        scaled += mask
        # scaled = scaled.permute(0, 2, 1, 3)
        # scaled = scaled.permute(1, 0, 2, 3) + mask + abbas
        # scaled = scaled.permute(0, 2, 1, 3)


    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, value)
    return values






class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, model_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        self.qkv_generator = nn.Linear(input_dim, 3*model_dim)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, x, mask=None):
        # print("--testing inside the model--")
        batch_size, sequence_len, _ = x.shape
        qkv = self.qkv_generator(x)
        # print(f'shape of qkv : {qkv.shape}')
        qkv = qkv.reshape(batch_size, self.n_heads, sequence_len, 3*self.head_dim)
        # qkv = qkv.permute(0, 2, 1, 3)
        # print(f'shape of qkv after reshaping: {qkv.shape}')
        q, k, v = qkv.chunk(3, dim=-1)
        # print(f'shape of q : {q.shape}')
        # print(f'shape of k : {k.shape}')
        # print(f'shape of v : {v.shape}')
        values = apply_attention(q, k, v, mask)
        # print(f'values after attention : {values.shape}')
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_len, self.n_heads * self.head_dim)    
        # print(f'values after reshaping : {values.shape}')
        out = self.final_linear(values)
        # print(f'shape of final output : {out.shape}')
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
    print("--testing multihead attention class--")
    input_dim = 20
    d_model = 64
    num_heads = 4

    batch_size = 3
    sequence_length = 5
    x = torch.randn( (batch_size, sequence_length, input_dim) )
    # print(f'input shape: {x.shape}')
    model = MultiHeadAttention(x.shape[-1], d_model, num_heads)
    out = model.forward(x)

