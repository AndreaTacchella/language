import TextualData.TextualData
import LSTM.lstm as lstm
import numpy as np
import torch.optim as optim
import torch
from torch.autograd import Variable
from utils import generate_pos
import time

hidden_size = 64
batch_size = 50
string_len = 50
valid_batches = 200
n_layers = 2
starting_lr = .025
lr_decay_factor = 2.
pos = 1
#Train the network to predict the charachter appearing after stride - normally 1
stride = 1


text = TextualData.TextualData.TextualData(path='data/full_shak_eng.txt', lower=True)
if pos == 1:
    print 'Computing POS...'
    text.compute_pos()
    print 'POS computed'
    print 'alpha len', text.alpha_len, 'pos len', text.pos_len, 'product', text.alpha_len*text.pos_len
    # print text.alphabet
    # for l in text.alphabet:
    #     print 'let', l


if pos == 1:
    rnn = lstm.LSTMmodel(hidden_s=hidden_size, input_s=text.alpha_len*text.pos_len, n_layers=n_layers)
else:
    rnn = lstm.LSTMmodel(hidden_s=hidden_size, input_s=text.alpha_len, n_layers=n_layers)
rnn.optimizer = optim.Adam(rnn.parameters(), lr=starting_lr)
print 'training set length:', text.train_len


print_every = 5
update_every = 50
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
        #maybe check validation loss
        if done_batches%update_every == 0:
            #compute valid loss
            previous_valid_loss = valid_loss
            rnn.init_hidden()
            valid_loss = np.mean([rnn.loss_func(rnn.forward(inp), tar) for inp, tar in
                                  text.get_valid_batch(string_len,valid_batches,0,stride,pos)]).data[0]
            torch.save(rnn.state_dict(), 'models/LSTM_nlay_'+str(n_layers)+'_hidsize_'+str(hidden_size)+'_pos_'
                       +str(pos)+'_itr_'+str(epochs*index+index)+'_loss_'+str('{:2.2f}'.format(valid_loss))+'.md')
            print 'valid loss', valid_loss, 'previous valid loss',previous_valid_loss
            #generate a sample
            rnn.init_hidden()
            test_b = text.get_test_batch(120, 1, 0, 1, 1)
            input = test_b[0][0]
            new_text, new_pos = generate_pos(rnn, text, input_tensor=input, verbose=False)
            print '/\\'*20
            print ''.join(new_text)
            print '/\\' * 20
            rnn.init_hidden()
            #maybe decrease learning rate
            if previous_valid_loss < valid_loss:
                starting_lr /= lr_decay_factor
                print 'decreasing learning rate to: ', starting_lr
                rnn.optimizer = optim.Adam(rnn.parameters(), lr=starting_lr)

print time.time()-t

