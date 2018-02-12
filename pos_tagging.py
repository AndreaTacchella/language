import TextualData.TextualData
import LSTM.lstm as lstm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import time
import spacy

nlp = spacy.load('en')

text = TextualData.TextualData.TextualData(path='data/full_shak_eng.txt', lines=10000)

text.compute_pos()

print text.pos_to_ix.keys()