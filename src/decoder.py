import torch
import torch.nn as nn
from decoder_layer import DecoderLayer
from sequential_decoder import SequentialDecoder
from sentence_embedding import SentenceEmbedding



class Decoder(nn.Module):
    def __init__(self, input_shape, model_dim, hidden_fc,
                  n_heads,
                  max_sequence_len, language2idx,
                  START_TOKEN, END_TOKEN, PADDING_TOKEN, 
                  drop_out_prob=0.1, n_layers=1):
        super(Decoder, self).__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_len, model_dim, language2idx,
                                                     START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(input_shape, model_dim, hidden_fc, n_heads, drop_out_prob)
                                          for _ in range(n_layers)])


    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y






if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    drop_prob = 0.1
    batch_size = 2
    max_sequence_length = 100
    hidden_fc = 2048
    num_layers = 2
    START_TOKEN = '<s>'
    END_TOKEN = '<\s>'
    PADDING_TOKEN = '<pad>'
    english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', '@', ';',
                        '[', '\\', ']',
                        '^', '_', '`', 
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                        'y', 'z', 
                        '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN
                        ]
    english_to_index = {v:k for k,v in enumerate(english_vocabulary)}
    y = ['hello my name is javad', 'its pleasure working with transformers']
    x = torch.randn( (batch_size, max_sequence_length, d_model) ).to(torch.device('cuda'))
    # y = torch.randn( (batch_size, max_sequence_length, d_model) ) 
    # mask = torch.full([max_sequence_length, max_sequence_length] , float('-inf'))
    # mask = torch.triu(mask, diagonal=1)
    decoder = Decoder(x.shape, d_model, hidden_fc, num_heads, max_sequence_length, english_to_index
                      , START_TOKEN, END_TOKEN, PADDING_TOKEN, drop_prob, num_layers).to(torch.device('cuda'))
    out = decoder(x, y, None, None, True ,True)
    print(out.shape)