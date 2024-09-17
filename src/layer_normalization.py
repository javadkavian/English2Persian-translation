import torch
import torch.nn  as nn





class LayerNormalization(nn.Module):
    def __init__(self, input_shape): #inputs shape should be like : (sequence_len, batch_size, embedding_size)
        super(LayerNormalization, self).__init__()
        self.parametes_shape = input_shape[-2:]
        self.gamma = nn.Parameter(torch.ones(self.parametes_shape))
        self.beta = nn.Parameter(torch.zeros(self.parametes_shape))


    def forward(self, x):
        dimensions = [-(i+1) for i in range(len(self.parametes_shape))] 
        mean = x.mean(dim = dimensions, keepdim=True)
        var = ((x - mean)**2).mean(dim=dimensions, keepdim=True)
        std = (var + 1e-5).sqrt()
        y = (x - mean) / std
        return self.gamma * y + self.beta




if __name__ == "__main__":
    batch_size = 5
    sentence_length = 10
    embedding_dim = 20
    inputs = torch.randn(sentence_length, batch_size, embedding_dim)
    print("--input shape befor layer normalization--")
    print(inputs.shape)
    layer = LayerNormalization(inputs.shape)
    print("--input shape after layer normalization--")
    print(layer(inputs).shape)
