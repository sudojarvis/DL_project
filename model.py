import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x):
        # x shape: (seq_len, N)
        embedding = self.embedding(x)
        # embedding shape: (seq_len, N, hidden_size)
        outputs, (hidden, cell) = self.lstm(embedding)
        # outputs shape: (seq_len, N, hidden_size)
        # hidden shape: (num_layers, N, hidden_size)
        # cell shape: (num_layers, N, hidden_size)
        return hidden, cell
    


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear((hidden_size * 2), hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden shape: (num_layers, N, hidden_size)
        # encoder_outputs shape: (src_len, N, hidden_size)
        src_len = encoder_outputs.shape[0]
        hidden = hidden[-1]
        # hidden shape: (N, hidden_size)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden shape: (N, src_len, hidden_size)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs shape: (N, src_len, hidden_size)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy shape: (N, src_len, hidden_size)
        attention = self.v(energy).squeeze(2)
        # attention shape: (N, src_len)
        return F.softmax(attention, dim=1)
    

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout, attention):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention = attention
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM((hidden_size * 2), hidden_size, num_layers, dropout=dropout)
        self.fc_out = nn.Linear((hidden_size * 2), output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell, encoder_outputs):
        # x shape: (N)
        # hidden shape: (num_layers, N, hidden_size)
        # cell shape: (num_layers, N, hidden_size)
        # encoder_outputs shape: (src_len, N, hidden_size)
        x = x.unsqueeze(0)
        # x shape: (1, N)
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, hidden_size)
        a = self.attention(hidden, encoder_outputs)
        # a shape: (N, src_len)
        a = a.unsqueeze(1)
        # a shape: (N, 1, src_len)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs shape: (N, src_len, hidden_size)
        weighted = torch.bmm(a, encoder_outputs)
        # weighted shape: (N, 1, hidden_size)
        weighted = weighted.permute(1, 0, 2)
        # weighted shape: (1, N, hidden_size)
        rnn_input = torch.cat((embedding, weighted), dim=2)
        # rnn_input shape: (1, N, hidden_size * 2)
        outputs, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)
        # hidden shape: (num_layers, N, hidden_size)
        # cell shape: (num_layers, N, hidden_size)
        predictions = self.fc_out(torch.cat((outputs, weighted), dim=2).squeeze(0))
        # predictions shape: (N, output_size)
        return predictions, hidden, cell
    

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # source shape: (src_len, N)
        # target shape: (trg_len, N)
        batch_size = target.shape[1]
        trg_len = target.shape[0]
        trg_vocab_size = self.decoder.output_size
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(source)
        # hidden shape: (num_layers, N, hidden_size)
        # cell shape: (num_layers, N, hidden_size)
        x = target[0]
        # x shape: (N)
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(x, hidden, cell, source)
            # output shape: (N, output_size)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            x = target[t] if teacher_force else top1
        return outputs
    



# from save_and_load import load_train_data, load_test_data
# train=list(load_train_data())
# test=load_test_data()


with open('train_source', 'r') as f:
    train_source = f.readlines()

tokenized_data=[]

with open('train_target', 'r') as f:
    train_target = f.readlines()



from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]



for sentences in train_source:
    for character in sentences:
        tokenized_data.append(character)

# print (tokenized_data)

from collections import Counter

token_counts = Counter([token for character in tokenized_data for token in character])

# print (len(token_counts))

token_to_index={token: index for index, token in enumerate(token_counts)}

# print (token_to_index)  

from torchtext.vocab import Vocab

vocab = Vocab(token_counts, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

# print (vocab['<pad>'])

# after conveting toekinzed data to index data we need to convert it to tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

seq_len=500
batch_size=32

# def collate_batch(batch):
#     label_list, text_list = [], []
#     for (_text, _label) in batch:
#         label_list.append(_label)
#         text_list.append(torch.tensor([vocab[token] for token in _text], dtype=torch.long))
#     label_list = torch.tensor(label_list, dtype=torch.long)
#     text_list = nn.utils.rnn.pad_sequence(text_list, padding_value=vocab['<pad>'])
#     return text_list, label_list

# train_dataset = CustomDataset(train_source, train_target)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

# print (train_loader.dataset[0])

# for i, (x, y) in enumerate(train_loader):
#     print (x)
#     print (y)
    break