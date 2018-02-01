# -*- coding: utf-8 -*-
from io import open
import numpy as np
from torch.autograd import Variable
import torch
import spacy

class TextualData():
    def __init__(self, path, lines=-1, lower=False):
        self.full_text = []
        self.train_text = []
        self.valid_text = []
        self.test_text = []
        self.alphabet = []
        self.alpha_len = 0
        self.letter_to_ix = {}
        self.lower = lower
        self.__import_text_file(path, lines = lines)
        self.train_len = len(self.train_text)
        self.valid_len = len(self.valid_text)
        self.test_len = len(self.test_text)
        self.vocab_len = 0
        self.word_to_ix = {}
        self.latest_char = 0
        #pos data from spacy
        self.pos_to_ix = {}
        self.ix_to_pos = {}
        self.full_pos = []
        self.pos_len = 0
        self.train_pos = []
        self.valid_pos = []
        self.test_pos = []


    def string_to_tensor(self, string):
        my_str = string
        #my_str += '/'
        ret = Variable(torch.zeros(len(my_str), 1, self.alpha_len))
        for i in range(len(my_str)):
            ret[i,0,self.letter_to_ix[my_str[i]]] = 1
        return ret


    # def string_to_tensor_pos(self, string, pos):
    #     if len(string) != len(pos):
    #         raise ValueError('Pos and String must be of the same length')
    #     my_str = string
    #     #my_str += '/'
    #     ret = Variable(torch.zeros(len(my_str), 1, self.alpha_len+self.pos_len))
    #     for i in range(len(my_str)):
    #         ret[i,0,self.letter_to_ix[my_str[i]]] = 1
    #         ret[i,0,self.alpha_len+pos[i]] = 1
    #     return ret

    def string_to_tensor_pos(self, string, pos):
        if len(string) != len(pos):
            raise ValueError('Pos and String must be of the same length')
        my_str = string
        #my_str += '/'
        ret = Variable(torch.zeros(len(my_str), 1, self.alpha_len*self.pos_len))
        for i in range(len(my_str)):
            ret[i,0,(pos[i]*self.alpha_len)+self.letter_to_ix[my_str[i]]] = 1
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

    def get_batch(self, string_len, batch_size, start, stride = 1, pos = 0):
        if start + string_len*batch_size >= self.train_len:
            raise IndexError('Not enough text to return this batch')
        batch = []
        st = start
        if pos == 0:
            for i in range(batch_size):
                batch.append(self.string_to_tensor(self.train_text[st:st + string_len]))
                st += string_len-1
            targets = [torch.cat([self.onehot_to_class(b[0]) for b in batch[bn][stride:]]) for bn in range(batch_size)]
            batch = torch.cat(batch, dim=1)[:string_len-stride]
            return [[batch[:,i],targets[i]] for i in range(batch_size)]
        if pos == 1:
            for i in range(batch_size):
                batch.append(self.string_to_tensor_pos(self.train_text[st:st + string_len], self.train_pos[st:st + string_len]))
                st += string_len-1
            targets = [torch.cat([self.onehot_to_class(b[0]) for b in batch[bn][stride:]]) for bn in range(batch_size)]
            batch = torch.cat(batch, dim=1)[:string_len-stride]
            return [[batch[:,i],targets[i]] for i in range(batch_size)]

    def get_valid_batch(self, string_len, batch_size, start, stride = 1, pos = 0):
        if start + string_len*batch_size >= self.valid_len:
            raise IndexError('Not enough text to return this batch')
        batch = []
        st = start
        if pos == 0:
            for i in range(batch_size):
                batch.append(self.string_to_tensor(self.valid_text[st:st + string_len]))
                st += string_len-1
            targets = [torch.cat([self.onehot_to_class(b[0]) for b in batch[bn][stride:]]) for bn in range(batch_size)]
            batch = torch.cat(batch, dim=1)[:string_len-stride]
            return [[batch[:,i],targets[i]] for i in range(batch_size)]
        if pos == 1:
            for i in range(batch_size):
                batch.append(self.string_to_tensor_pos(self.valid_text[st:st + string_len], self.valid_pos[st:st + string_len]))
                st += string_len-1
            targets = [torch.cat([self.onehot_to_class(b[0]) for b in batch[bn][stride:]]) for bn in range(batch_size)]
            batch = torch.cat(batch, dim=1)[:string_len-stride]
            return [[batch[:,i],targets[i]] for i in range(batch_size)]

    def get_test_batch(self, string_len, batch_size, start, stride = 1, pos = 0):
        if start + string_len*batch_size >= self.test_len:
            raise IndexError('Not enough text to return this batch')
        batch = []
        st = start
        if pos == 0:
            for i in range(batch_size):
                batch.append(self.string_to_tensor(self.test_text[st:st + string_len]))
                st += string_len-1
            targets = [torch.cat([self.onehot_to_class(b[0]) for b in batch[bn][stride:]]) for bn in range(batch_size)]
            batch = torch.cat(batch, dim=1)[:string_len-stride]
            return [[batch[:,i],targets[i]] for i in range(batch_size)]
        if pos == 1:
            for i in range(batch_size):
                batch.append(self.string_to_tensor_pos(self.test_text[st:st + string_len], self.test_pos[st:st + string_len]))
                st += string_len-1
            targets = [torch.cat([self.onehot_to_class(b[0]) for b in batch[bn][stride:]]) for bn in range(batch_size)]
            batch = torch.cat(batch, dim=1)[:string_len-stride]
            return [[batch[:,i],targets[i]] for i in range(batch_size)]


    def get_random_batch(self, string_len, batch_size, stride = 1):
        batch = [self.string_to_tensor(self.random_string_fixed_size(string_len)) for i in range(batch_size)]
        targets = [torch.cat([self.onehot_to_class(b[0]) for b in batch[bn][stride:]]) for bn in range(batch_size)]
        batch = torch.cat(batch, dim=1)[:string_len-stride]
        return [[batch[:,i],targets[i]] for i in range(batch_size)]

    def get_random_valid_batch(self, string_len, batch_size, stride = 1, pos = 0):
        rnd = np.random.randint(self.valid_len - batch_size)
        if pos == 0:
            batch = [self.string_to_tensor(self.valid_text[rnd:rnd+string_len]) for i in range(batch_size)]
            targets = [torch.cat([self.onehot_to_class(b[0]) for b in batch[bn][stride:]]) for bn in range(batch_size)]
            batch = torch.cat(batch, dim=1)[:string_len-stride]
            return [[batch[:,i],targets[i]] for i in range(batch_size)]
        if pos == 1:
            batch = [self.string_to_tensor_pos(self.valid_text[rnd:rnd+string_len], self.valid_pos[rnd:rnd+string_len])
                     for i in range(batch_size)]
            targets = [torch.cat([self.onehot_to_class(b[0]) for b in batch[bn][stride:]]) for bn in range(batch_size)]
            batch = torch.cat(batch, dim=1)[:string_len - stride]
            return [[batch[:, i], targets[i]] for i in range(batch_size)]

    def onehot_to_class(self, vector):
        return Variable(torch.LongTensor([i for i in range(len(vector.data)) if (vector.data[i] > 0)]))


    def __import_text_file(self, path, lines = -1, valid_size = .2, test_size = .2) :
        self.full_text = open(path, encoding="utf-8").read()
        self.full_text = self.full_text.replace(u"\u2019",'\'')
        self.full_text = self.full_text.replace(u"\u2018", '\'')
        self.full_text = self.full_text.replace(u"\u2013", '-')
        self.full_text = self.full_text.replace(u"\u2014", '-')
        self.full_text = self.full_text.replace(u"\u2003", ' ')
        self.full_text = self.full_text.replace(u"\u201c", '\'')
        self.full_text = self.full_text.replace(u"\u201d", '\'')
        self.full_text = self.full_text.replace(u"\xef", 'i')
        self.full_text = self.full_text.replace(u"\u201a", ',')
        self.full_text = self.full_text.replace(u"\xe9", 'e')
        self.full_text = self.full_text.replace(u"\u2026", '...')
        self.full_text = self.full_text.replace("&", '')
        self.full_text = self.full_text.replace(u"\xa0", ' ')
        self.full_text = self.full_text.replace(u"\xe7", 'c')
        self.full_text = self.full_text.replace(u"\xe0", 'a')

        if self.lower is True:
            self.full_text = self.full_text.lower()
        if lines > 0:
            self.full_text = self.full_text[:lines]

        self.train_text = self.full_text[:-int((valid_size+test_size)*len(self.full_text))]
        self.valid_text = self.full_text[-int((valid_size+test_size)*len(self.full_text)):-int(test_size*len(self.full_text))]
        self.test_text = self.full_text[-int(test_size*len(self.full_text)):]
        self.alphabet = list(set([l for l in self.full_text]))
        self.alpha_len = len(self.alphabet)
        self.letter_to_ix = dict((self.alphabet[i], i) for i in range(self.alpha_len))

    def compute_pos(self):
        nlp = spacy.load('en')
        doc = nlp(self.full_text[:300000])
        allpos = set([tok.pos_ for tok in doc])
        self.pos_to_ix = {}
        self.ix_to_pos = {}
        ix = 0
        for pos in allpos:
            self.pos_to_ix[pos] = ix
            self.ix_to_pos[ix] = pos
            ix += 1
        mylen = 100000
        begin = 0
        end = mylen
        while begin < len(self.full_text):
            mystr = self.full_text[begin:end]
            begin = end
            end = min(end + mylen, len(self.full_text))
            doc = nlp(mystr)
            pos_by_char = []
            for tok in doc:
                pos_by_char.append([self.pos_to_ix[tok.pos_] for l in tok.text_with_ws])
            pos_by_char = np.array([item for sublist in pos_by_char for item in sublist])
            self.full_pos.append(pos_by_char)
            #print begin, end, len(pos_by_char)
        self.full_pos = [item for sublist in self.full_pos for item in sublist]
        self.train_pos = self.full_pos[:self.train_len]
        self.valid_pos = self.full_pos[self.train_len:self.train_len + self.valid_len]
        self.test_pos = self.full_pos[-self.test_len:]
        print 'Created POS data. Train_pos len:', len(self.train_pos), 'Valid_pos len:', len(self.valid_pos)
        self.pos_len = len(self.pos_to_ix.keys())


