from torch import nn
from torch.autograd import Variable
import torch
import numpy as np

class LSTMmodel(nn.Module):
    def __init__(self, input_s, hidden_s, n_layers = 2, gpu = False):
        super(LSTMmodel, self).__init__()
        self.gpu = gpu
        self.n_layers = n_layers
        self.hidden_size = hidden_s
        self.input_size = input_s
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=self.n_layers, dropout=0.25)
        self.linear = nn.Linear(self.hidden_size, self.input_size)
        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        if torch.__version__ == '0.2.0_4':
            self.logsoft = nn.LogSoftmax()
        else:
            self.logsoft = nn.LogSoftmax(dim=0)
        self.optimizer = 0.
        self.loss_func =  nn.NLLLoss()
        self.h0 = 0
        self.c0 = 0
        self.init_hidden()


    def forward(self, inp):
        #self.init_hidden()
        true_inp = []
        for i in range(len(inp)):
            true_inp.append(self.embedding(inp[i]))
        true_inp = torch.stack(true_inp)
        out, (self.h0, self.c0) = self.lstm(true_inp, (self.h0, self.c0))
        self.h0 = Variable(self.h0.data)
        self.c0 = Variable(self.c0.data)
        if self.gpu is True:
            self.h0=self.h0.cuda()
            self.c0=self.c0.cuda()
        out = out.view(-1, self.hidden_size)
        true_out = []
        # print 'len inp', len(inp)
        for i in range(len(inp)):
            true_out.append(self.logsoft(self.linear(out[i])))
        # print len(true_out)
        # print true_out[0].size()
        return torch.stack(true_out)

    def train(self, batch):
        self.zero_grad()
        loss = np.mean([self.loss_func(self.forward(inp), tar) for inp,tar in batch])
        torch.nn.utils.clip_grad_norm(self.parameters(), 2.)
        #loss = self.loss_func(self.forward(input), labels)
        loss.backward()
        self.optimizer.step()
        return loss

    # def forward_known_hidden(self, inp, h):
    #     #(h0,c0) = h
    #     true_inp = []
    #     for i in range(len(inp)):
    #         true_inp.append(self.embedding(inp[i]))
    #     true_inp = torch.stack(true_inp)
    #     out, h0 = self.lstm(true_inp, h)
    #     out = out.view(-1, self.hidden_size)
    #     true_out = []
    #     for i in range(len(inp)):
    #         true_out.append(self.logsoft(self.linear(out[i])))
    #     return torch.stack(true_out), h0


    def init_hidden(self):
        self.h0 = Variable(torch.randn(self.n_layers, 1, self.hidden_size))
        self.c0 = Variable(torch.randn(self.n_layers, 1, self.hidden_size))
        if self.gpu is True:
            self.h0=self.h0.cuda()
            self.c0=self.c0.cuda()

    # def init_hidden(self):
    #     self.h0 = torch.randn(self.n_layers, 1, self.hidden_size)
    #     self.c0 = torch.randn(self.n_layers, 1, self.hidden_size)

    def set_hidden(self, (h0,c0)):
        (self.h0, self.c0) = (h0,c0)
        if self.gpu is True:
            self.h0=self.h0.cuda()
            self.c0=self.c0.cuda()
