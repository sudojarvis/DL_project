import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
# import torchtext.utils.tensorboard as SummaryWriter
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

with open('train_target', 'r') as f:
    train_target = f.readlines()


with open('test_source', 'r') as f:
    test_source = f.readlines()

with open('test_target', 'r') as f:
    test_target = f.readlines()


from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]



def tokenize_source(text):
    tokenized_data_train = []
    for sentences in train_source:
        for character in sentences:
            tokenized_data_train.append(character)
    return tokenized_data_train

# tokenized_data_train = []
# for sentences in train_source:
#     for character in sentences:
#         tokenized_data_train.append(character)
# return tokenized_data_train

# print (tokenized_data)

def vocab_(tokenized_data_):
    from collections import Counter
    token_counts = Counter([token for character in tokenized_data_ for token in character])
    vocab=token_counts.keys()
    return vocab

# print (len(vocab_(tokenized_data_train)))

# token_counts_train = Counter([token for character in tokenized_data_train for token in character])
def tokenize_target(text):
    tokenized_data_target = []
    for sentences in train_target:
        for character in sentences:
            tokenized_data_target.append(character)
    return tokenized_data_target

# tokenized_data_target = []
# for sentences in train_target:
#     for character in sentences:
#         tokenized_data_target.append(character)

# print (tokenized_data)

# from collections import Counter

# token_counts_target = Counter([token for character in tokenized_data_target for token in character])


# print (len(token_counts))

# token_to_index={token: index for index, token in enumerate(token_counts)}

# # print (token_to_index)  

from torchtext.vocab import Vocab

# vocab = Vocab(vocab_(tokenized_data_train), min_freq=1, specials=['<unk>', '<pad>', '<sos>', '<eos>'])


# print (vocab['<pad>'])

# after conveting toekinzed data to index data we need to convert it to tensor

from torchtext.data import Field, BucketIterator

source_= Field(tokenize=tokenize_source,init_token='<sos>',eos_token='<eos>',lower=True)
target_ = Field(tokenize=tokenize_target,init_token='<sos>',eos_token='<eos>',lower=True)
# print (source_.tokenize(train_source[0]))
source_.build_vocab(train_source,min_freq=1)
target_.build_vocab(train_target,min_freq=1)

print(len(source_.vocab))
class Transformer(nn.Module):
    def __init__(self, embedding_size,src_vocab_size,src_pad_idx,num_heads,num_encoder_layer,num_decoder_layer,forword_expansion,dropout,max_len,device):
        super(Transformer,self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size,embedding_size)
        self.src_position_embedding = nn.Embedding(max_len,embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len,embedding_size)
        self.device = device
        self.transformer = nn.Transformer(embedding_size,num_heads,num_encoder_layer,num_decoder_layer,forword_expansion,dropout)
        self.fc_out = nn.Linear(embedding_size,src_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
    def make_src_mask(self,src):
        src_mask = src.transpose(0,1) == self.src_pad_idx
        return src_mask.to(self.device)
    def forward(self,src,trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape
        src_positions = (torch.arange(0,src_seq_length).unsqueeze(1).expand(src_seq_length,N).to(self.device))
        trg_positions = (torch.arange(0,trg_seq_length).unsqueeze(1).expand(trg_seq_length,N).to(self.device))
        embed_src = self.dropout((self.src_word_embedding(src) + self.src_position_embedding(src_positions)))
        embed_trg = self.dropout((self.src_word_embedding(trg) + self.src_position_embedding(trg_positions)))
        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)
        out = self.transformer(embed_src,embed_trg,src_key_padding_mask=src_padding_mask,tgt_mask=trg_mask)
        return self.fc_out(out)
    

# print (source_.tokenize(train_source[0][0]  ))

# print(source_.build_vocab(train_source, min_freq=2))
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



src_vocab_size = len(source_.vocab)
trg_vocab_size = len(target_.vocab)



print (src_vocab_size)
print (trg_vocab_size)



embedding_size = 512
num_heads = 8
num_encoder_layer = 3
num_decoder_layer = 3
dropout = 0.10
max_len = 500
forword_expansion = 512

src_pad_idx = source_.vocab.stoi['<pad>']

# writer=SummaryWriter('runs/loss_plot')

from sklearn.model_selection import train_test_split

trian_source1, val_source1, train_target1, val_target1 = train_test_split(train_source, train_target, test_size=0.2, random_state=42)

# print (val_data[0])

train_data=CustomDataset(trian_source1,train_target1)
val_data=CustomDataset(val_source1,val_target1)

model=Transformer(embedding_size,src_vocab_size,src_pad_idx,num_heads,num_encoder_layer,num_decoder_layer,forword_expansion,dropout,max_len,device).to(device)
optimizer=optim.Adam(model.parameters(),lr=0.0001)
pad_idx = target_.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)



train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for batch_idx,batch in enumerate(train_loader):
    print(batch[0])
    src=list(batch[0])
    trg=list(batch[1])
    output=model(src,trg[:-1,:])
    output_dim=output.shape[-1]
    output=output.reshape(-1,output_dim)
    trg=trg[1:].reshape(-1)
    optimizer.zero_grad()
    loss=criterion(output,trg)
    loss.backward()
    optimizer.step()
    print (loss.item())

# def train(model, iterator, optimizer, criterion, clip):
#     model.train()
#     epoch_loss = 0
#     for i, batch in enumerate(iterator):
#         src = batch[0].to(device)
#         trg = batch[1].to(device)
#         optimizer.zero_grad()
#         output = model(src, trg[:-1,:])
#         output_dim = output.shape[-1]
#         output = output.reshape(-1, output_dim)
#         trg = trg[1:].reshape(-1)
#         loss = criterion(output, trg)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#         optimizer.step()
#         epoch_loss += loss.item()
#     return epoch_loss / len(iterator)


# def evaluate(model, iterator, criterion):

#     model.eval()
#     epoch_loss = 0
#     with torch.no_grad():
#         for i, batch in enumerate(iterator):
#             src = batch[0].to(device)
#             trg = batch[1].to(device)
#             output = model(src, trg[:-1,:])
#             output_dim = output.shape[-1]
#             output = output.reshape(-1, output_dim)
#             trg = trg[1:].reshape(-1)
#             loss = criterion(output, trg)
#             epoch_loss += loss.item()
#     return epoch_loss / len(iterator)


# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

# num_epochs = 10
# clip = 1
# best_valid_loss = float('inf')
# for epoch in range(num_epochs):
#     train_loss = train(model, train_loader, optimizer, criterion, clip)
#     valid_loss = evaluate(model, val_loader, criterion)
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'model.pt')
#     print(f'Epoch {epoch+1}: | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

# model.load_state_dict(torch.load('model.pt'))
# test_data=CustomDataset(test_source,test_target)




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import random

# seq_len=500
# batch_size=32

# # def collate_batch(batch):
# #     label_list, text_list = [], []
# #     for (_text, _label) in batch:
# #         label_list.append(_label)
# #         text_list.append(torch.tensor([vocab[token] for token in _text], dtype=torch.long))
# #     label_list = torch.tensor(label_list, dtype=torch.long)
# #     text_list = nn.utils.rnn.pad_sequence(text_list, padding_value=vocab['<pad>'])
# #     return text_list, label_list

# # train_dataset = CustomDataset(train_source, train_target)
# # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

# # print (train_loader.dataset[0])

# # for i, (x, y) in enumerate(train_loader):
# #     print (x)
# #     print (y)
#     break