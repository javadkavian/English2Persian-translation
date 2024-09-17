import torch
import torch.nn  as nn





class LayerNormalization(nn.Module):
    def __init__(self, input_shape): #inputs shape should be like : (sequence_len, batch_size, embedding_size)
        super(LayerNormalization, self).__init__()
        self.parametes_shape = input_shape[-2:]
        self.gamma = nn.Parameter(torch.ones(self.parametes_shape))
        self.beta = nn.Parameter(torch.zeros(self.parametes_shape))


    def forward(self, x):
        # print("--testing inside the class--")
        dimensions = [-(i+1) for i in range(len(self.parametes_shape))] 
        # print(f'dimensions : {dimensions}')
        mean = x.mean(dim = dimensions, keepdim=True)
        # print(f'shape of mean : {mean.shape}')
        var = ((x - mean)**2).mean(dim=dimensions, keepdim=True)
        # print(f'shape of variance : {var.shape}')
        std = (var + 1e-5).sqrt()
        # print(f'shape of std : {std.shape}')
        y = (x - mean) / std
        # print(f'shape of y : {y.shape}')
        out = self.gamma * y + self.beta
        # print(f'shape of output : {out.shape}')
        return out




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
