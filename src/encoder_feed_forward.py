import torch
import torch.nn as nn
import torch.nn.functional as F






class EncoderFeedForward(nn.Module):
    def __init__(self, model_dim, hidden_fc, drop_out_prob=0.1):
        super(EncoderFeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_fc)
        self.fc2 = nn.Linear(hidden_fc, model_dim)
        self.dropout = nn.Dropout(p=drop_out_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x 






if __name__ == "__main__":
    x = torch.randn((1, 4, 512))
    model = EncoderFeedForward(512, 800)
    print(model(x).shape)