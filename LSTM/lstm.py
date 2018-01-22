from torch import nn
from torch.autograd import Variable
import torch

class LSTMmodel(nn.Module):
    def __init__(self, input_s, hidden_s):
        super(LSTMmodel, self).__init__()
        self.n_layers = 2
        self.hidden_size = hidden_s
        self.input_size = input_s
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=self.n_layers, dropout=0.5)
        self.linear = nn.Linear(self.hidden_size, self.input_size)
        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        self.logsoft = nn.LogSoftmax()
        self.optimizer = 0.
        self.loss_func =  nn.NLLLoss()

    def forward(self, inp):
        h0 = Variable(torch.randn(self.n_layers, 1, self.hidden_size))
        c0 = Variable(torch.randn(self.n_layers, 1, self.hidden_size))
        true_inp = []
        for i in range(len(inp)):
            true_inp.append(self.embedding(inp[i]))
        true_inp = torch.stack(true_inp)
        out, h0 = self.lstm(true_inp, (h0, c0))
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
        input = batch[0]
        labels = batch[1]
        loss = self.loss_func(self.forward(input), labels)
        loss.backwards()
        self.optimizer.step()
        return loss








    def init_hidden(self):
        return (Variable(torch.randn(1, self.hidden_size)),
                Variable(torch.randn(1, self.hidden_size)))