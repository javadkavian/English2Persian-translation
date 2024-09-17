import torch
import torch.nn as nn




class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.max_sequence_len = max_seq_len
        self.model_dim = model_dim

    def forward(self):    
        denominator = torch.pow(1000, (torch.arange(0, self.model_dim, 2, dtype=torch.float32) / self.model_dim))
        position = torch.arange(self.max_sequence_len).reshape(self.max_sequence_len, 1)
        even = torch.sin(position / denominator)
        odd = torch.cos(position / denominator)
        PE = torch.stack([even, odd], dim=2).flatten(start_dim=1, end_dim=2)
        return PE
    


if __name__ == "__main__":
    pe = PositionalEncoding(5, 20)
    print(pe.forward())    
