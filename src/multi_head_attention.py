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


