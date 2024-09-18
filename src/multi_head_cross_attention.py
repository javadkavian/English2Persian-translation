import torch
import torch.nn as nn
from multi_head_attention import apply_attention




class MultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dim, n_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        self.kv_generator = nn.Linear(model_dim, 2*model_dim)
        self.q_generator = nn.Linear(model_dim, model_dim)
        self.fc = nn.Linear(model_dim, model_dim)


    def forward(self, x, y, mask=None):
        batch_size, sequence_len, _ = x.shape
        kv = self.kv_generator(x)
        kv = kv.reshape(batch_size, sequence_len, self.n_heads, 2*self.head_dim).permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        q = self.q_generator(y)
        q = q.reshape(batch_size, sequence_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        values = apply_attention(q, k, v, mask)
        values = values.reshape(batch_size, sequence_len, self.model_dim)
        out = self.fc(values)
        return out
    




if __name__ == "__main__":
    batch_size = 30
    n_heads = 8
    model_dim = 64
    input_dim = 64
    max_sequence_len = 20
    x = torch.randn((batch_size, max_sequence_len, input_dim))
    y = torch.randn((batch_size, max_sequence_len, input_dim))
    model = MultiHeadCrossAttention(model_dim, n_heads)
    out = model(x, y)
    print(out.shape)