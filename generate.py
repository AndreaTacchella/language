import TextualData.TextualData
import LSTM.lstm as lstm
import numpy as np
import torch.optim as optim
from utils import generate_pos
import torch
import time

hidden_size = 256
batch_size = 50
string_len = 50
valid_batches = 200
n_layers = 2
starting_lr = .025
lr_decay_factor = 2.
pos = 1
#Train the network to predict the charachter appearing after stride - Leave = 1
stride = 1
gpu = False


text = TextualData.TextualData.TextualData(path='data/full_shak_eng.txt', lower=True)
if pos == 1:
    print 'Computing POS...'
    text.compute_pos()
    print 'POS computed'
    print 'alpha len', text.alpha_len, 'pos len', text.pos_len, 'product', text.alpha_len*text.pos_len
# print text.pos_len
# print text.get_batch(string_len, batch_size, 0, 1, 1)

if pos == 1:
    rnn = lstm.LSTMmodel(hidden_s=hidden_size, input_s=text.alpha_len*text.pos_len, n_layers=n_layers, gpu=gpu)
else:
    rnn = lstm.LSTMmodel(hidden_s=hidden_size, input_s=text.alpha_len, n_layers=n_layers, gpu=gpu)

rnn.load_state_dict(torch.load('models/LSTM_nlay_2_hidsize_256_pos_1_itr_1000000_loss_1.66.md'))
# print text.get_valid_batch(3,2,0,1,1)
test_b = text.get_test_batch(120, 1, 0, 1, 1)
input = test_b[0][0]
new_text, new_pos = generate_pos(rnn, text, input_tensor=input, verbose=False)
print '/\\' * 20
print ''.join(new_text)
print '/\\' * 20
