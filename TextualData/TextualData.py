from io import open
import numpy as np
from torch.autograd import Variable
import torch

class TextualData():
    def __init__(self, path):
        self.full_text = []
        self.alphabet = []
        self.alpha_len = 0
        self.letter_to_ix = {}
        self.__import_text_file(path)
        self.text_len = len(self.full_text)
        self.vocab_len = 0
        self.word_to_ix = {}



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
            my_rnd = np.random.randint(self.text_len-batch_len)
        else:
            my_rnd = rnd
        while (self.full_text[my_rnd] != " ") & (my_rnd > 0):
            my_rnd = my_rnd - 1
        k = 0
        while (self.full_text[my_rnd + batch_len + k] != " ") & (my_rnd + batch_len + k < self.text_len):
            k = k + 1
        input_string = self.full_text[my_rnd:my_rnd + batch_len + k]
        return input_string

    def random_string_fixed_size(self, batch_len, rnd=-1):
        if rnd < 0:
            my_rnd = np.random.randint(self.text_len - batch_len)
        else:
            my_rnd = rnd
        input_string = self.full_text[my_rnd:my_rnd + batch_len]
        return input_string

    def get_batch(self, string_len, batch_size):
        batch = [self.string_to_tensor(self.random_string_fixed_size(string_len)) for i in range(len(batch_size))]
        return torch.cat(batch,axis=1)

    def __import_text_file(self, path):
        self.full_text = open(path, encoding="utf-8").read()
        self.alphabet = list(set([l for l in self.full_text]))
        self.alpha_len = len(self.alphabet)
        self.letter_to_ix = dict((self.alphabet[i], i) for i in range(self.alpha_len))