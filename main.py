import TextualData.TextualData
import LSTM.lstm as lstm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import time
from torch.autograd import Variable

text = TextualData.TextualData.TextualData(path='data/full_shak_eng.txt')

def generate(model, inp='#', temp = 0.7, my_len = 50):
    row = inp
    tensor_string = text.string_to_tensor(row)
    hidden = model.init_hidden()
    for i in range(len(tensor_string)):
        output, hidden = model.forward_known_hidden(tensor_string[i], hidden)
    row += text.alphabet[torch.multinomial(output.data.view(-1).div(temp).exp(),1)[0]]
    for i in range(my_len):
        output, hidden = model.forward_known_hidden(text.string_to_tensor(row)[-1], hidden)
        row += text.alphabet[torch.multinomial(output.data.view(-1).div(temp).exp(),1)[0]]
    return row

hidden_size = 512
#rnn = LSTMmodel(alpha_len, hidden_size)
rnn = lstm.LSTMmodel(hidden_s=hidden_size, input_s=text.alpha_len, n_layers=3)
rnn.optimizer = optim.Adam(rnn.parameters(), lr=0.004)
batch_size = 50
string_len = 50

print_every = 10
tot_loss=0
t = time.time()
for i in range(1, 5001):
    my_loss = rnn.train(text.get_batch(string_len,batch_size,1))
    tot_loss += my_loss.data[0]
    if i%print_every == 0:
        print i, '='*20, tot_loss/print_every
        tot_loss = 0
        print generate(rnn)
        torch.save(rnn.state_dict(), 'models/LSTM_'+str(i)+'.md')
print time.time()-t