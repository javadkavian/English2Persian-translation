import torch
import torch.nn as nn
from encoder_layer import EncoderLayer
from sentence_embedding import SentenceEmbedding
from sequential_encoder import SequentialEncoder



class Encoder(nn.Module):
    
    def __init__(self, input_shape, model_dim, hidden_fc,
                  n_heads ,drop_out_prob, n_layers,
                  max_sequence_len,
                  language2idx,
                  START_TOKEN,
                  END_TOKEN,
                  PADDING_TOKEN):
        super(Encoder, self).__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_len, model_dim, language2idx, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.encoder_layers = SequentialEncoder(*[EncoderLayer(input_shape, model_dim, hidden_fc, n_heads, drop_out_prob)
                                            for _ in range(n_layers)])
        

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.encoder_layers(x, self_attention_mask)
        return x





if __name__ == "__main__":
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
    
    d_model = 512
    num_heads = 8
    drop_prob = 0.1
    batch_size = 30
    max_sequence_length = 200
    ffn_hidden = 2048
    num_layers = 5
    # x = torch.randn( (batch_size, max_sequence_length, d_model) ) 
    english_to_index = {v:k for k,v in enumerate(english_vocabulary)}
    x = ['hello my name is javad', 'its pleasure working with transformers']
    encoder = Encoder((30, 200, d_model), d_model, ffn_hidden, num_heads,
                       drop_prob, num_layers, max_sequence_length,
                       english_to_index, START_TOKEN, 
                       END_TOKEN, PADDING_TOKEN).to(torch.device('cuda'))
    out = encoder(x, None, True, True)
    print(out.shape)