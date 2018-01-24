from io import open
import numpy as np
from torch.autograd import Variable
import torch

class TextualData():
    def __init__(self, path):
        self.full_text = []
        self.train_text = []
        self.valid_text = []
        self.test_text = []
        self.alphabet = []
        self.alpha_len = 0
        self.letter_to_ix = {}
        self.__import_text_file(path)
        self.train_len = len(self.train_text)
        self.valid_len = len(self.valid_text)
        self.test_len = len(self.test_text)
        self.vocab_len = 0
        self.word_to_ix = {}
        self.latest_char = 0



    def string_to_tensor(self, string):
        my_str = string
        #my_str += '/'
        ret = Variable(torch.zeros(len(my_str), 1, self.alpha_len))
        for i in range(len(my_str)):
            ret[i,0,self.letter_to_ix[my_str[i]]] = 1
        return ret

    def words_to_tensor(self, string):
        my_str = string
        #my_str += '/'
        ret = Variable(torch.zeros(len(my_str), 1, self.vocab_len))
        for i in range(len(my_str)):
            if (my_str[i] in self.word_to_ix.keys()):
                ret[i,0,self.word_to_ix[my_str[i]]] = 1
            else:
                ret[i,0,self.word_to_ix['UNK']] = 1
        return ret

    def random_string(self, batch_len, rnd=-1):
        if rnd < 0:
            my_rnd = np.random.randint(self.train_len - batch_len)
        else:
            my_rnd = rnd
        while (self.train_text[my_rnd] != " ") & (my_rnd > 0):
            my_rnd = my_rnd - 1
        k = 0
        while (self.train_text[my_rnd + batch_len + k] != " ") & (my_rnd + batch_len + k < self.train_len):
            k = k + 1
        input_string = self.train_text[my_rnd:my_rnd + batch_len + k]
        return input_string

    def random_string_fixed_size(self, batch_len, rnd=-1):
        if rnd < 0:
            my_rnd = np.random.randint(self.train_len - batch_len)
        else:
            my_rnd = rnd
        input_string = self.train_text[my_rnd:my_rnd + batch_len]
        return input_string

    def random_valid_string_fixed_size(self, batch_len, rnd=-1):
        if rnd < 0:
            my_rnd = np.random.randint(self.valid_len - batch_len)
        else:
            my_rnd = rnd
        input_string = self.valid_text[my_rnd:my_rnd + batch_len]
        return input_string

    def get_batch(self, string_len, batch_size, start, stride = 1):
        if start + string_len*batch_size >= self.train_len:
            raise IndexError('Not enough text to return this batch')
        batch = []
        st = start
        for i in range(batch_size):
            batch.append(self.string_to_tensor(self.train_text[st:st + string_len]))
            st += string_len-1
        #batch = [self.string_to_tensor(self.random_string_fixed_size(string_len)) for i in range(batch_size)]
        targets = [torch.cat([self.onehot_to_class(b[0]) for b in batch[bn][stride:]]) for bn in range(batch_size)]
        batch = torch.cat(batch, dim=1)[:string_len-stride]
        return [[batch[:,i],targets[i]] for i in range(batch_size)]

    def get_random_batch(self, string_len, batch_size, stride = 1):
        batch = [self.string_to_tensor(self.random_string_fixed_size(string_len)) for i in range(batch_size)]
        targets = [torch.cat([self.onehot_to_class(b[0]) for b in batch[bn][stride:]]) for bn in range(batch_size)]
        batch = torch.cat(batch, dim=1)[:string_len-stride]
        return [[batch[:,i],targets[i]] for i in range(batch_size)]

    def get_random_valid_batch(self, string_len, batch_size, stride = 1):
        batch = [self.string_to_tensor(self.random_valid_string_fixed_size(string_len)) for i in range(batch_size)]
        targets = [torch.cat([self.onehot_to_class(b[0]) for b in batch[bn][stride:]]) for bn in range(batch_size)]
        batch = torch.cat(batch, dim=1)[:string_len-stride]
        return [[batch[:,i],targets[i]] for i in range(batch_size)]

    #TODO: ADD label return in get_batch, so that batch is in the format expected by train method of model

    def onehot_to_class(self, vector):
        return Variable(torch.LongTensor([i for i in range(len(vector.data)) if vector.data[i] > 0]))

    def __import_text_file(self, path, valid_size = 50000, test_size = 50000) :
        self.full_text = open(path, encoding="utf-8").read()
        self.train_text = self.full_text[:-(valid_size+test_size)]
        self.valid_text = self.full_text[-(valid_size + test_size):-test_size]
        self.test_text = self.full_text[-test_size:]
        self.alphabet = list(set([l for l in self.train_text]))
        self.alpha_len = len(self.alphabet)
        self.letter_to_ix = dict((self.alphabet[i], i) for i in range(self.alpha_len))
