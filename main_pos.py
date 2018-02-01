import TextualData.TextualData
import LSTM.lstm as lstm
import numpy as np
import torch.optim as optim
import torch
import time

def generate(model, inp='#', temp = 0.7, my_len = 150):
    row = inp
    tensor_string = text.string_to_tensor(row)
    model.init_hidden()
    for i in range(len(tensor_string)):
        output = model.forward(tensor_string[i])
    row += text.alphabet[torch.multinomial(output.data.view(-1).div(temp).exp(),1)[0]]
    for i in range(my_len):
        output= model.forward(text.string_to_tensor(row)[-1])
        row += text.alphabet[torch.multinomial(output.data.view(-1).div(temp).exp(),1)[0]]
    return row

hidden_size = 16
batch_size = 50
string_len = 50
valid_batches = 200
n_layers = 2
starting_lr = .025
lr_decay_factor = 2.
pos = 1
#Train the network to predict the charachter appearing after stride - Leave = 1
stride = 1


text = TextualData.TextualData.TextualData(path='data/full_shak_eng.txt', lines = 200000, lower=True)
if pos == 1:
    print 'Computing POS...'
    text.compute_pos()
    print 'POS computed'
    print 'alpha len', text.alpha_len, 'pos len', text.pos_len, 'product', text.alpha_len*text.pos_len
    print text.alphabet
    for le in text.alphabet:
        print 'char', le
# print text.pos_len
# print text.get_batch(string_len, batch_size, 0, 1, 1)


if pos == 1:
    rnn = lstm.LSTMmodel(hidden_s=hidden_size, input_s=text.alpha_len+text.pos_len,
                         output_s=text.alpha_len, n_layers=n_layers)
else:
    rnn = lstm.LSTMmodel(hidden_s=hidden_size, input_s=text.alpha_len,
                         output_s=text.alpha_len, n_layers=n_layers)
rnn.optimizer = optim.Adam(rnn.parameters(), lr=starting_lr)
print 'training set length:', text.train_len

print_every = 1
update_every = 2
tot_loss=0
t = time.time()
valid_loss = np.mean([rnn.loss_func(rnn.forward(inp), tar) for inp, tar in
                      text.get_valid_batch(string_len,valid_batches,0,stride,pos)]).data[0]
print 'starting valid loss', valid_loss

for epochs in range(100):
    index = 0
    done_batches = 0
    rnn.init_hidden()
    while (index < text.train_len-string_len*batch_size) & (starting_lr > 0.00005):

        my_loss = rnn.train(text.get_batch(string_len,batch_size,index,stride,pos))
        index += string_len*batch_size
        done_batches += 1
        tot_loss += my_loss.data[0]
        if done_batches%print_every == 0:
            print epochs, '-', '{:2.2f}'.format(1.0*done_batches/(text.train_len/(string_len*batch_size))), '%',\
                '=' *  int(10 *tot_loss / print_every), tot_loss / print_every
            tot_loss = 0

        if done_batches%update_every == 0:
            previous_valid_loss = valid_loss
            rnn.init_hidden()
            valid_loss = np.mean([rnn.loss_func(rnn.forward(inp), tar) for inp, tar in
                                  text.get_valid_batch(string_len,valid_batches,0,stride,pos)]).data[0]
            torch.save(rnn.state_dict(), 'models/LSTM_nlay_'+str(n_layers)+'_hidsize_'+str(hidden_size)+'_pos_'
                       +str(pos)+'_itr_'+str(epochs*index+index)+'_loss_'+str('{:2.2f}'.format(valid_loss))+'.md')
            print 'valid loss', valid_loss, 'previous valid loss',previous_valid_loss
            if previous_valid_loss < valid_loss:
                starting_lr /= lr_decay_factor
                print 'decreasing learning rate to: ', starting_lr
                rnn.optimizer = optim.Adam(rnn.parameters(), lr=starting_lr)
            rnn.init_hidden()
            #print generate(rnn)
print time.time()-t
