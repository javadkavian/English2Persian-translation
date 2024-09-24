import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding


import pandas as pd


def tokenize(sentence,
              start_token,
                end_token,
                  language2idx,
                    max_sequence_len,
                      START_TOKEN,
                        END_TOKEN,
                          PADDING_TOKEN):
    word_indices = [language2idx[token] for token in list(sentence)]
    if len(word_indices) > max_sequence_len - 2:
        word_indices = word_indices[:max_sequence_len - 2]
    if start_token:
        word_indices.insert(0, language2idx[START_TOKEN])
    if end_token:
        word_indices.append(language2idx[END_TOKEN])
    for _ in range(len(word_indices), max_sequence_len):
        word_indices.append(language2idx[PADDING_TOKEN])
    return torch.tensor(word_indices)            




class SentenceEmbedding(nn.Module):

    def __init__(self, max_sequence_len, model_dim, language2idx, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language2idx)
        self.max_sequence_len = max_sequence_len
        self.embedding = nn.Embedding(self.vocab_size, model_dim)
        self.language2idx = language2idx
        self.positional_encoding = PositionalEncoding(model_dim, max_sequence_len)
        self.dropout = nn.Dropout(p=.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.device = torch.device('cuda')

    def batch_tokenize(self, batch, start_token, end_token):
        tokenized = []
        for idx in range(len(batch)):
            tokenized.append(tokenize(batch[idx], 
                                      start_token, 
                                      end_token, 
                                      self.language2idx, 
                                      self.max_sequence_len, 
                                      self.START_TOKEN, 
                                      self.END_TOKEN, 
                                      self.PADDING_TOKEN)) 

            

        tokenized = torch.stack(tokenized) 
        return tokenized.to(self.device)


    def forward(self, x, start_token, end_token):
        x = self.batch_tokenize(x, start_token, end_token) 
        num_embeddings = self.embedding.num_embeddings
        if (x >= num_embeddings).any() or (x < 0).any():
            print("Invalid indices found in input:", x)
        x = self.embedding(x)
        pos = self.positional_encoding().to(self.device)
        x = self.dropout(x + pos)
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
    
    persian_vocabulary = [
    START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ';',
    ':', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', 
    'آ', 'ا', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'ژ', 'س', 'ش', 
    'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و', 'ه', 'ی',
    'ء', 'ۀ', 'ؤ', 'ي', 'ك', 'ة', '‌', 'ٔ', 'ى', PADDING_TOKEN, END_TOKEN
    ]









    def helper_english(x:str):
        for c in x:
            if not c in english_vocabulary:
                x = x.replace(c, '')
        return x

    def helper_persian(x:str):
        for c in x:
            if not c in persian_vocabulary:
                x = x.replace(c, '')
        return x                


    index_to_persian = {k:v for k,v in enumerate(persian_vocabulary)}
    persian_to_index = {v:k for k,v in enumerate(persian_vocabulary)}
    index_to_english = {k:v for k,v in enumerate(english_vocabulary)}
    english_to_index = {v:k for k,v in enumerate(english_vocabulary)}
    df = pd.read_csv('././dataset/shortened_dataset.csv')
    df['english'] = df['english'].apply(str.lower)
    df['english'] = df['english'].apply(helper_english)
    df['persian'] = df['persian'].apply(helper_persian)
    persian_sentences = df['persian'].to_list()
    english_sentences = df['english'].to_list()
    model = SentenceEmbedding(100, 512, persian_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN).to(torch.device('cuda'))
    out = model(persian_sentences, True, True)
    print(out.shape)


