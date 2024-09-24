import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder



class Transformer(nn.Module):
    def __init__(self, input_shape, model_dim, hidden_fc, n_heads, drop_out_prob,
                 n_layers, max_sequence_len, persian_vocab_size, english2idx, persian2idx,
                 START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_shape, model_dim, hidden_fc, n_heads, drop_out_prob, n_layers,
                               max_sequence_len, english2idx, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        
        self.decoder = Decoder(input_shape, model_dim, hidden_fc, n_heads, max_sequence_len, persian2idx, 
                               START_TOKEN, END_TOKEN, PADDING_TOKEN, drop_out_prob, n_layers)
        
        self.fc = nn.Linear(model_dim, persian_vocab_size)


    def forward(self, x, y, encoder_self_attention_mask = None, 
                            decoder_self_attention_mask = None,
                            decoder_cross_attention_mask = None,
                            encoder_start_token = False,
                            encoder_end_token = False,
                            decoder_start_token = False,
                            decoder_end_token = False):
        x = self.encoder(x, encoder_self_attention_mask, encoder_start_token, encoder_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask,
                            decoder_cross_attention_mask,
                              decoder_start_token, decoder_end_token) 
        out = self.fc(out)
        return out
    



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
    
    persian_vocabulary = [
    START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ';',
    ':', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', 
    'آ', 'ا', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'ژ', 'س', 'ش', 
    'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و', 'ه', 'ی',
    'ء', 'ۀ', 'ؤ', 'ي', 'ك', 'ة', '‌', 'ٔ', 'ى', PADDING_TOKEN, END_TOKEN
    ]
    index_to_persian = {k:v for k,v in enumerate(persian_vocabulary)}
    persian_to_index = {v:k for k,v in enumerate(persian_vocabulary)}
    index_to_english = {k:v for k,v in enumerate(english_vocabulary)}
    english_to_index = {v:k for k,v in enumerate(english_vocabulary)}
    y = ['سلام من جوادم', 'به معماری ترنسفورمر خیلی علاقه دارم']
    x = ['hello i am javad', 'i love transformers so much']
    batch_size = 2
    max_sequence_len = 100
    model_dim = 512
    hidden_fc = 1024
    n_heads = 8
    model = Transformer((batch_size, max_sequence_len, model_dim), model_dim, hidden_fc,
                        n_heads, 0.1, 2, max_sequence_len, len(persian_vocabulary), english_to_index, persian_to_index,
                        START_TOKEN, END_TOKEN, PADDING_TOKEN).to(torch.device('cuda'))
    
    output = model(x, y)
    print(output.shape)


