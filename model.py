import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math


trainsource=[]

with open('train_source', 'r') as f:
   
   for line in f:
        
        trainsource.append(line.strip())
       

traintarget=[]
with open('train_target', 'r') as f:
    # train_target = f.readlines()
    for line in f:
        traintarget.append(line.strip())



testsource=[]
with open('test_source', 'r') as f:

    for line in f:
        testsource.append(line.strip())



test_target=[]
with open('test_target', 'r') as f:
    # test_target = f.readlines()
    for line in f:
        test_target.append(line.strip())



# print(len(trainsource))
# print(len(traintarget))


vocab_source=set()
def tokenize(spec_list):
    tokenized=[]
    for sub_list in spec_list:
        char_lst=[]
        for char in sub_list:
            char_lst.append(char)
        tokenized.append(char_lst)

    return tokenized


train_source=tokenize(trainsource)
# print(train_source[0])


def vocab_(tokenized_data_):
    vocab=[]
    for sub_list in tokenized_data_:
        for char in sub_list:          
            if char not in vocab:
                vocab.append(char)

    return vocab


train_source_vocab=vocab_(train_source)
target_vocab=vocab_(traintarget)


input_sequences=[]
for seq in train_source:
    input_seq=[train_source_vocab.index(char) for char in seq]
    input_sequences.append(input_seq)

target_seqeunces=[]
for seq in traintarget:
    target_seq=[target_vocab.index(char) for char in seq]
    target_seqeunces.append(target_seq)

# print(input_sequences)

# padded_input_sequences = torch.nn.utils.rnn.pad_sequence(
#     [torch.tensor(seq) for seq in input_sequences],
#     batch_first=True
# )



max_len = 374
padded_input_sequences = torch.nn.utils.rnn.pad_sequence(
    [torch.tensor(seq + [0]*(max_len - len(seq))) for seq in input_sequences],
    batch_first=True,
    padding_value=0
)

print(padded_input_sequences.shape)

# padded_output_sequences = torch.nn.utils.rnn.pad_sequence(
#     [torch.tensor(seq) for seq in target_seqeunces],
#     batch_first=True
# define the maximum length

padded_output_sequences = torch.nn.utils.rnn.pad_sequence(
    [torch.tensor(seq + [0]*(max_len - len(seq))) for seq in target_seqeunces],  # truncate to maximum length
    batch_first=True,
    padding_value=0  # set padding value to 0
)

print("padded_output_sequences.shape: ", padded_output_sequences.shape)


testsources_seqeunces=[]
for seq in testsource:
    source_seq=[train_source_vocab.index(char) for char in seq]
    testsources_seqeunces.append(source_seq)


padded_test_sources_sequences = torch.nn.utils.rnn.pad_sequence(
    [torch.tensor(seq + [0]*(max_len - len(seq))) for seq in testsources_seqeunces],  # truncate to maximum length
    batch_first=True,
    padding_value=0  # set padding value to 0
)




test_target_seqeunces=[]
for seq in test_target:
    target_seq=[target_vocab.index(char) for char in seq]
    test_target_seqeunces.append(target_seq)


padded_test_target_sequences = torch.nn.utils.rnn.pad_sequence(
    [torch.tensor(seq + [0]*(max_len - len(seq))) for seq in test_target_seqeunces],  # truncate to maximum length
    batch_first=True,
    padding_value=0  # set padding value to 0
)

# print("padded_test_target_sequences.shape: ", padded_test_target_sequences.shape)

class Data(torch.utils.data.Dataset):
    
    def __init__(self, input_sequences, output_sequences):
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences
    
    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        return self.input_sequences[index], self.output_sequences[index]
    


dataset = Data(padded_input_sequences, padded_output_sequences)
train_size=int(0.8*len(dataset))
val_size=len(dataset)-train_size
train_data, val_data=torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader=torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader=torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True) 

test_dataset = Data(padded_test_sources_sequences, padded_test_target_sequences)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)








class encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super().__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.embedding=nn.Embedding(input_size, embedding_size)
        self.rnn=nn.LSTM(embedding_size, hidden_size, 512, dropout=p)
        self.dropout=nn.Dropout(p)
        
    def forward(self, x):
        embedding=self.dropout(self.embedding(x))
        outputs, (hidden, cell)=self.rnn(embedding)
        return hidden, cell
    

# input_size_encoder=len(train_source_vocab)
# embedding_size=100
# hidden_size=256
# num_layers=2
# p=0.5
# encoder_net=encoder(input_size_encoder, embedding_size, hidden_size, num_layers, p)
# print(encoder_net)


class decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super().__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.embedding=nn.Embedding(input_size, embedding_size)
        self.rnn=nn.LSTM(embedding_size, hidden_size, 512, dropout=p)
        self.fc_hidden=nn.Linear(hidden_size, hidden_size)
        self.fc_output=nn.Linear(hidden_size, output_size)
        self.dropout=nn.Dropout(p)
        
    def forward(self, x, hidden, cell):
        x=x.unsqueeze(0)
        embedding=self.dropout(self.embedding(x))
        outputs, (hidden, cell)=self.rnn(embedding, (hidden, cell))
        hidden=self.fc_hidden(hidden)
        outputs=self.fc_output(outputs)
        predictions=outputs.squeeze(0)
        return predictions, hidden, cell
    




input_size_encoder=len(train_source_vocab)
embedding_size=500
hidden_size=512
num_layers=2
p=0.5
encoder_net=encoder(input_size_encoder, embedding_size, hidden_size, num_layers, p)

input_size_decoder=len(target_vocab)
output_size=len(target_vocab)
decoder_net=decoder(input_size_decoder, embedding_size, hidden_size, output_size, num_layers, p)




class seq2seq(nn.Module):
    def __init__(self, input_size_encoder, embedding_size, hidden_size, output_size, num_layers_encoder,num_layers_decoder, p, device):
        super().__init__()
        self.encoder=encoder(input_size_encoder, embedding_size, hidden_size, num_layers_encoder, p).to(device)
        self.decoder=decoder(output_size, embedding_size, hidden_size, output_size, num_layers_decoder, p).to(device)
        self.device=device
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size=source.shape[1]
        #target_len=target.shape[0]
        target_len=target.shape[0]
        target_vocab_size=len(target_vocab)
        
        outputs=torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        hidden, cell=self.encoder(source)
        
        x=target[0]
        for t in range(1, target_len):
            output, hidden, cell=self.decoder(x, hidden, cell)
            outputs[t]=output
            best_guess=output.argmax(1)
            x=target[t] if random.random()<teacher_forcing_ratio else best_guess
        return outputs



model=seq2seq(input_size_encoder, embedding_size, hidden_size, output_size, num_layers, num_layers, p, device='cpu')

optimizer=torch.optim.Adam(model.parameters(), lr=0.001)

criterion=nn.CrossEntropyLoss(ignore_index=0)

from torch.nn.utils.rnn import pad_sequence
n_epochs=1
for epoch in range(n_epochs):
    train_loss=0
    model.train()
    for i, batch in enumerate(train_loader):
        # Pad the input sequences to the same length
        source = pad_sequence(batch[0], batch_first=True)
        target = pad_sequence(batch[1], batch_first=True)
        
        output = model(source, target)
        # print(output)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        target = target[1:].reshape(-1)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f}')





def test(model, test_loader, criterion, target_vocab):
    test_loss = 0
    translations = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Pad the input sequences to the same length
            source = pad_sequence(batch[0], batch_first=True)
            target = pad_sequence(batch[1], batch_first=True)
            
            output = model(source, target, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            target = target[1:].reshape(-1)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            # Compute the predicted translation
            pred_translation = []
            for timestep in output:
                pred_token = timestep.argmax().item()
                pred_translation.append(target_vocab[pred_token])
            translations.append(' '.join(pred_translation))
    
    avg_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    print("Example translations are: ", translations[:5])
    return translations

translations = test(model,val_loader, criterion, target_vocab)



































# class Encoder(nn.Module):
#     def __init__(self, input_size, emb_dim,hid_dim, n_layers, dropout):
#         super(Encoder, self).__init__()
#         self.hidden_size = hid_dim 
#         self.num_layers = n_layers
#         self.dropout = dropout
#         self.embedding = nn.Embedding(input_size, emb_dim)
#         self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # x shape: (seq_len, N)
#         embedded = self.dropout(self.embedding(x))
        
#         # embedding shape: (seq_len, N, hidden_size)
#         outputs, (hidden, cell) = self.lstm(embedded)
#         # outputs shape: (seq_len, N, hidden_size)
#         # hidden shape: (num_layers, N, hidden_size)
#         # cell shape: (num_layers, N, hidden_size)
#         return hidden, cell
    

# class Decoder(nn.Module):
#     def __init__(self, output_size, emb_dim, hid_dim, n_layers, dropout):
#         super(Decoder, self).__init__()
#         self.hidden_size = hid_dim
#         self.num_layers = n_layers
#         self.dropout = dropout
#         self.embedding = nn.Embedding(output_size, emb_dim)
#         self.lstm = nn.LSTM(emb_dim, hid_dim , n_layers, dropout=dropout)
#         self.fc = nn.Linear(hid_dim, output_size)
#         self.dropout = nn.Dropout(dropout)
#         self.output_size = output_size

#     def forward(self, x, hidden, cell):
#         input = x.unsqueeze(0)
#         # input shape: (1, N)
#         embedded = self.dropout(self.embedding(input))
#         output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
#         # output shape: (1, N, hidden_size)
#         prediction = self.fc(output.squeeze(0))
#         # prediction shape: (N, length_target_vocabulary)
#         return prediction, hidden, cell


# class seq2seq(nn.Module):
#     def __init__(self, encoder,decoder, device):
#         super().__init__()
#         # self.encoder=Encoder(input_size_encoder, embedding_size, hidden_size, num_layers_encoder, p).to(device)
#         # self.decoder=Decoder(output_size, embedding_size, hidden_size, output_size, num_layers_decoder, p).to(device)

#         self.encoder=encoder
#         self.decoder=decoder
#         self.device=device
        
#     def forward(self, source, target, teacher_forcing_ratio=0.5):
#         batch_size=source.shape[1]
#         target_len=target.shape[0]
#         print("target", target.shape)
#         print("source", source.shape)
#         target_len=target.shape[0]
#         target_vocab_size=self.decoder.output_size   #error
        
#         outputs=torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
#         print("output", outputs.shape)
#         hidden, cell=self.encoder(source)
#         print("output", outputs.shape)
#         print("hidden", hidden.shape)
#         print("cell", cell.shape)
#         x=target[0,:]
#         hidden, cell=self.encoder(source)
#         for t in range(1, target_len):
#             output, hidden, cell=self.decoder(x, hidden, cell)
           
            
#             outputs[t]=output
#             best_guess=output.argmax(1)
#             x=target[t] if random.random()<teacher_forcing_ratio else best_guess
#         return outputs


# input_size=len(train_source_vocab)
# output_size=len(target_vocab)
# enc_emb_dim=256
# dec_emb_dim=256
# hid_dim=512
# enc_layers=2
# dec_layers=2
# enc_dropout=0.5
# dec_dropout=0.5

# enc=Encoder(input_size, enc_emb_dim, hid_dim, enc_layers, enc_dropout)
# dec=Decoder(output_size, dec_emb_dim, hid_dim, dec_layers, dec_dropout)
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model=seq2seq(enc, dec, device).to(device)

# optimizer=torch.optim.Adam(model.parameters(), lr=0.001)

# criterion=nn.CrossEntropyLoss(ignore_index=0)

# from torch.nn.utils.rnn import pad_sequence

# n_epochs=20
# for epoch in range(n_epochs):
#     train_loss=0
#     model.train()
#     for i, batch in enumerate(train_loader):
#         # Pad the input sequences to the same length
#         source = batch[0]
#         print('source_shape', source.shape)
        
#         target = batch[1]
        
#         output = model(source, target)
#         print(output)
#         output_dim = output.shape[-1]
#         output = output[1:].view(-1, output_dim)
#         target = target[1:].reshape(-1)
#         loss = criterion(output, target)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
#         optimizer.step()
#         optimizer.zero_grad()
#         train_loss += loss.item()
#         print(train_loss)







# print(padded_input_sequences)
# print("--------------------------")
# print(padded_output_sequences)

# n_train=int(len(train_source)*0.8)

# train_data = [(padded_input_sequences[:n_train], padded_output_sequences[:n_train])]
# valid_data = [(padded_input_sequences[n_train:], padded_output_sequences[n_train:])]


# class Seq2Seq(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Seq2Seq, self).__init__()
#         self.input_size = input_size  # Save the input size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
#         self.decoder = nn.GRU(output_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x, y):
#         batch_size = x.size(0)  # Get the batch size dynamically
#         input_size = self.input_size  # Get the input size dynamically
#         _, h = self.encoder(x.view(batch_size, -1, input_size))
#         out, _ = self.decoder(y, h)
#         out = self.fc(out)
#         return out

# # Define the loss function and optimizer
# model = Seq2Seq(len(target_vocab), 128, len(target_vocab))
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# n_epochs = 10
# # Train the model
# for epoch in range(n_epochs):
#     for input_seq, output_seq in train_data:
#         optimizer.zero_grad()
#         output = model(input_seq, output_seq[:, :-1])
#         loss = criterion(output.view(-1, len(target_vocab)), output_seq[:, 1:].reshape(-1))
#         loss.backward()
#         optimizer.step()
#         print(loss)





# # Define the model architecture
# # class Seq2Seq(nn.Module):
# #     def __init__(self, input_size, output_size, hidden_size):
# #         super(Seq2Seq, self).__init__()
# #         self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
# #         self.decoder = nn.GRU(output_size, hidden_size, batch_first=True)
# #         self.fc = nn.Linear(hidden_size, output_size)

# #     def forward(self, x, y):
# #         _, h = self.encoder(x)
# #         out, _ = self.decoder(y, h)
# #         out = self.fc(out)
# #         return out

# # # Define the loss function and optimizer
# # model = Seq2Seq(len(train_source_vocab), len(target_vocab), 128)
# # criterion = nn.CrossEntropyLoss()
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# # n_epochs = 10

# # # Train the model
# # for epoch in range(n_epochs):
# #     for input_seq, output_seq in train_data:
# #         optimizer.zero_grad()
# #         output = model(input_seq, output_seq[:, :-1])
# #         loss = criterion(output.view(-1, len(target_vocab)), output_seq[:, 1:].contiguous().view(-1))
# #         loss.backward()
# #         optimizer.step()
# #     print(f"Epoch {epoch+1}, Loss: {loss.item()}")  # print loss at the end of each epoch





# # Evaluate the model on the validation set
# with torch.no_grad():
#     total_loss = 0
#     for input_seq, output_seq in valid_data:
#         output = model(input_seq, output_seq[:,:-1])
#         loss = criterion(output.view(-1, len(vocab)), output_seq[:,1:].contiguous().view(-1))
#         total_loss += loss.item()
#     print("Validation

# from torch.utils.data import Dataset, DataLoader

# class CustomDataset(Dataset):
#     def __init__(self, source, target):
#         self.source = source
#         self.target = target

#     def __len__(self):
#         return len(self.source)

#     def __getitem__(self, idx):
#         return self.source[idx], self.target[idx]





# def tokenize_source(text):
#     tokenized_data_train = []
#     for sentences in train_source:
#         sub_list=[]
#         for character in sentences:
#             print(character)
#             sub_list.append(character)
#         sub_list.append(sub_list)
#     return tokenized_data_train



# def vocab_(tokenized_data_):
#     from collections import Counter
#     token_counts = Counter([token for character in tokenized_data_ for token in character])
#     vocab=token_counts.keys()
#     return vocab

# # print (len(vocab_(tokenized_data_train)))

# # token_counts_train = Counter([token for character in tokenized_data_train for token in character])
# def tokenize_target(text):
#     tokenized_data_target = []
#     for sentences in train_target:
#         sub_list=[]
#         for character in sentences:
#             sub_list.append(character)
#         tokenized_data_target.append(sub_list)
#     return tokenized_data_target

# sen_tokenize_source=tokenize_source(train_source)
# sen_tokenize_target=tokenize_target(train_target)

# # vocab_source=vocab_(sen_tokenize_source)
# # vocab_target=vocab_(sen_tokenize_target)

# print(sen_tokenize_source)


# # def make_int(sen)














































































# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import random
# import math
# # import torchtext.utils.tensorboard as SummaryWriter
# class Encoder(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, dropout):
#         super(Encoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout)

#     def forward(self, x):
#         # x shape: (seq_len, N)
#         embedding = self.embedding(x)
#         # embedding shape: (seq_len, N, hidden_size)
#         outputs, (hidden, cell) = self.lstm(embedding)
#         # outputs shape: (seq_len, N, hidden_size)
#         # hidden shape: (num_layers, N, hidden_size)
#         # cell shape: (num_layers, N, hidden_size)
#         return hidden, cell
    


# class Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.attn = nn.Linear((hidden_size * 2), hidden_size)
#         self.v = nn.Linear(hidden_size, 1, bias=False)

#     def forward(self, hidden, encoder_outputs):
#         # hidden shape: (num_layers, N, hidden_size)
#         # encoder_outputs shape: (src_len, N, hidden_size)
#         src_len = encoder_outputs.shape[0]
#         hidden = hidden[-1]
#         # hidden shape: (N, hidden_size)
#         hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
#         # hidden shape: (N, src_len, hidden_size)
#         encoder_outputs = encoder_outputs.permute(1, 0, 2)
#         # encoder_outputs shape: (N, src_len, hidden_size)
#         energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
#         # energy shape: (N, src_len, hidden_size)
#         attention = self.v(energy).squeeze(2)
#         # attention shape: (N, src_len)
#         return F.softmax(attention, dim=1)
    

# class Decoder(nn.Module):
#     def __init__(self, output_size, hidden_size, num_layers, dropout, attention):
#         super(Decoder, self).__init__()
#         self.output_size = output_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.attention = attention
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.lstm = nn.LSTM((hidden_size * 2), hidden_size, num_layers, dropout=dropout)
#         self.fc_out = nn.Linear((hidden_size * 2), output_size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, hidden, cell, encoder_outputs):
#         # x shape: (N)
#         # hidden shape: (num_layers, N, hidden_size)
#         # cell shape: (num_layers, N, hidden_size)
#         # encoder_outputs shape: (src_len, N, hidden_size)
#         x = x.unsqueeze(0)
#         # x shape: (1, N)
#         embedding = self.dropout(self.embedding(x))
#         # embedding shape: (1, N, hidden_size)
#         a = self.attention(hidden, encoder_outputs)
#         # a shape: (N, src_len)
#         a = a.unsqueeze(1)
#         # a shape: (N, 1, src_len)
#         encoder_outputs = encoder_outputs.permute(1, 0, 2)
#         # encoder_outputs shape: (N, src_len, hidden_size)
#         weighted = torch.bmm(a, encoder_outputs)
#         # weighted shape: (N, 1, hidden_size)
#         weighted = weighted.permute(1, 0, 2)
#         # weighted shape: (1, N, hidden_size)
#         rnn_input = torch.cat((embedding, weighted), dim=2)
#         # rnn_input shape: (1, N, hidden_size * 2)
#         outputs, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
#         # outputs shape: (1, N, hidden_size)
#         # hidden shape: (num_layers, N, hidden_size)
#         # cell shape: (num_layers, N, hidden_size)
#         predictions = self.fc_out(torch.cat((outputs, weighted), dim=2).squeeze(0))
#         # predictions shape: (N, output_size)
#         return predictions, hidden, cell
    

# class Seq2Seq(nn.Module):

#     def __init__(self, encoder, decoder, device):
#         super(Seq2Seq, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.device = device

#     def forward(self, source, target, teacher_forcing_ratio=0.5):
#         # source shape: (src_len, N)
#         # target shape: (trg_len, N)
#         batch_size = target.shape[1]
#         trg_len = target.shape[0]
#         trg_vocab_size = self.decoder.output_size
#         outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
#         hidden, cell = self.encoder(source)
#         # hidden shape: (num_layers, N, hidden_size)
#         # cell shape: (num_layers, N, hidden_size)
#         x = target[0]
#         # x shape: (N)
#         for t in range(1, trg_len):
#             output, hidden, cell = self.decoder(x, hidden, cell, source)
#             # output shape: (N, output_size)
#             outputs[t] = output
#             teacher_force = random.random() < teacher_forcing_ratio
#             top1 = output.argmax(1)
#             x = target[t] if teacher_force else top1
#         return outputs
    



# # from save_and_load import load_train_data, load_test_data
# # train=list(load_train_data())
# # test=load_test_data()


# with open('train_source', 'r') as f:
#     train_source = f.readlines()

# with open('train_target', 'r') as f:
#     train_target = f.readlines()


# with open('test_source', 'r') as f:
#     test_source = f.readlines()

# with open('test_target', 'r') as f:
#     test_target = f.readlines()


# from torch.utils.data import Dataset, DataLoader

# class CustomDataset(Dataset):
#     def __init__(self, source, target):
#         self.source = source
#         self.target = target

#     def __len__(self):
#         return len(self.source)

#     def __getitem__(self, idx):
#         return self.source[idx], self.target[idx]



# def tokenize_source(text):
#     tokenized_data_train = []
#     for sentences in train_source:
#         for character in sentences:
#             tokenized_data_train.append(character)
#     return tokenized_data_train

# # tokenized_data_train = []
# # for sentences in train_source:
# #     for character in sentences:
# #         tokenized_data_train.append(character)
# # return tokenized_data_train

# # print (tokenized_data)

# def vocab_(tokenized_data_):
#     from collections import Counter
#     token_counts = Counter([token for character in tokenized_data_ for token in character])
#     vocab=token_counts.keys()
#     return vocab

# # print (len(vocab_(tokenized_data_train)))

# # token_counts_train = Counter([token for character in tokenized_data_train for token in character])
# def tokenize_target(text):
#     tokenized_data_target = []
#     for sentences in train_target:
#         for character in sentences:
#             tokenized_data_target.append(character)
#     return tokenized_data_target

# # tokenized_data_target = []
# # for sentences in train_target:
# #     for character in sentences:
# #         tokenized_data_target.append(character)

# # print (tokenized_data)

# # from collections import Counter

# # token_counts_target = Counter([token for character in tokenized_data_target for token in character])


# # print (len(token_counts))

# # token_to_index={token: index for index, token in enumerate(token_counts)}

# # # print (token_to_index)  

# from torchtext.vocab import Vocab

# # vocab = Vocab(vocab_(tokenized_data_train), min_freq=1, specials=['<unk>', '<pad>', '<sos>', '<eos>'])


# # print (vocab['<pad>'])

# # after conveting toekinzed data to index data we need to convert it to tensor

# from torchtext.data import Field, BucketIterator

# source_= Field(tokenize=tokenize_source,init_token='<sos>',eos_token='<eos>',lower=True)
# target_ = Field(tokenize=tokenize_target,init_token='<sos>',eos_token='<eos>',lower=True)
# # print (source_.tokenize(train_source[0]))
# source_.build_vocab(train_source,min_freq=1)
# target_.build_vocab(train_target,min_freq=1)

# print(len(source_.vocab))
# class Transformer(nn.Module):
#     def __init__(self, embedding_size,src_vocab_size,src_pad_idx,num_heads,num_encoder_layer,num_decoder_layer,forword_expansion,dropout,max_len,device):
#         super(Transformer,self).__init__()
#         self.src_word_embedding = nn.Embedding(src_vocab_size,embedding_size)
#         self.src_position_embedding = nn.Embedding(max_len,embedding_size)
#         self.trg_position_embedding = nn.Embedding(max_len,embedding_size)
#         self.device = device
#         self.transformer = nn.Transformer(embedding_size,num_heads,num_encoder_layer,num_decoder_layer,forword_expansion,dropout)
#         self.fc_out = nn.Linear(embedding_size,src_vocab_size)
#         self.dropout = nn.Dropout(dropout)
#         self.src_pad_idx = src_pad_idx
#     def make_src_mask(self,src):
#         src_mask = src.transpose(0,1) == self.src_pad_idx
#         return src_mask.to(self.device)
#     def forward(self,src,trg):
#         src_seq_length, N = src.shape
#         trg_seq_length, N = trg.shape
#         src_positions = (torch.arange(0,src_seq_length).unsqueeze(1).expand(src_seq_length,N).to(self.device))
#         trg_positions = (torch.arange(0,trg_seq_length).unsqueeze(1).expand(trg_seq_length,N).to(self.device))
#         embed_src = self.dropout((self.src_word_embedding(src) + self.src_position_embedding(src_positions)))
#         embed_trg = self.dropout((self.src_word_embedding(trg) + self.src_position_embedding(trg_positions)))
#         src_padding_mask = self.make_src_mask(src)
#         trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)
#         out = self.transformer(embed_src,embed_trg,src_key_padding_mask=src_padding_mask,tgt_mask=trg_mask)
#         return self.fc_out(out)
    

# # print (source_.tokenize(train_source[0][0]  ))

# # print(source_.build_vocab(train_source, min_freq=2))
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# src_vocab_size = len(source_.vocab)
# trg_vocab_size = len(target_.vocab)



# print (src_vocab_size)
# print (trg_vocab_size)



# embedding_size = 512
# num_heads = 8
# num_encoder_layer = 3
# num_decoder_layer = 3
# dropout = 0.10
# max_len = 500
# forword_expansion = 512

# src_pad_idx = source_.vocab.stoi['<pad>']

# # writer=SummaryWriter('runs/loss_plot')

# from sklearn.model_selection import train_test_split

# trian_source1, val_source1, train_target1, val_target1 = train_test_split(train_source, train_target, test_size=0.2, random_state=42)

# # print (val_data[0])

# train_data=CustomDataset(trian_source1,train_target1)
# val_data=CustomDataset(val_source1,val_target1)

# model=Transformer(embedding_size,src_vocab_size,src_pad_idx,num_heads,num_encoder_layer,num_decoder_layer,forword_expansion,dropout,max_len,device).to(device)
# optimizer=optim.Adam(model.parameters(),lr=0.0001)
# pad_idx = target_.vocab.stoi["<pad>"]
# criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)



# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

# import numpy as np
# import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# for batch_idx,batch in enumerate(train_loader):
#     print(batch[0])
#     src=list(batch[0])
#     trg=list(batch[1])
#     output=model(src,trg[:-1,:])
#     output_dim=output.shape[-1]
#     output=output.reshape(-1,output_dim)
#     trg=trg[1:].reshape(-1)
#     optimizer.zero_grad()
#     loss=criterion(output,trg)
#     loss.backward()
#     optimizer.step()
#     print (loss.item())

# # def train(model, iterator, optimizer, criterion, clip):
# #     model.train()
# #     epoch_loss = 0
# #     for i, batch in enumerate(iterator):
# #         src = batch[0].to(device)
# #         trg = batch[1].to(device)
# #         optimizer.zero_grad()
# #         output = model(src, trg[:-1,:])
# #         output_dim = output.shape[-1]
# #         output = output.reshape(-1, output_dim)
# #         trg = trg[1:].reshape(-1)
# #         loss = criterion(output, trg)
# #         loss.backward()
# #         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
# #         optimizer.step()
# #         epoch_loss += loss.item()
# #     return epoch_loss / len(iterator)


# # def evaluate(model, iterator, criterion):

# #     model.eval()
# #     epoch_loss = 0
# #     with torch.no_grad():
# #         for i, batch in enumerate(iterator):
# #             src = batch[0].to(device)
# #             trg = batch[1].to(device)
# #             output = model(src, trg[:-1,:])
# #             output_dim = output.shape[-1]
# #             output = output.reshape(-1, output_dim)
# #             trg = trg[1:].reshape(-1)
# #             loss = criterion(output, trg)
# #             epoch_loss += loss.item()
# #     return epoch_loss / len(iterator)


# # train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# # val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

# # num_epochs = 10
# # clip = 1
# # best_valid_loss = float('inf')
# # for epoch in range(num_epochs):
# #     train_loss = train(model, train_loader, optimizer, criterion, clip)
# #     valid_loss = evaluate(model, val_loader, criterion)
# #     if valid_loss < best_valid_loss:
# #         best_valid_loss = valid_loss
# #         torch.save(model.state_dict(), 'model.pt')
# #     print(f'Epoch {epoch+1}: | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

# # model.load_state_dict(torch.load('model.pt'))
# # test_data=CustomDataset(test_source,test_target)




# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torch.optim as optim
# # import random

# # seq_len=500
# # batch_size=32

# # # def collate_batch(batch):
# # #     label_list, text_list = [], []
# # #     for (_text, _label) in batch:
# # #         label_list.append(_label)
# # #         text_list.append(torch.tensor([vocab[token] for token in _text], dtype=torch.long))
# # #     label_list = torch.tensor(label_list, dtype=torch.long)
# # #     text_list = nn.utils.rnn.pad_sequence(text_list, padding_value=vocab['<pad>'])
# # #     return text_list, label_list

# # # train_dataset = CustomDataset(train_source, train_target)
# # # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

# # # print (train_loader.dataset[0])

# # # for i, (x, y) in enumerate(train_loader):
# # #     print (x)
# # #     print (y)
# #     break
