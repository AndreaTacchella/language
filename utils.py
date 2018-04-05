import torch
from torch.autograd import Variable
import numpy as np

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

def generate_pos(model, text, input_tensor, temp=1., gen_len=10, verbose = False):
    tensor = input_tensor
    char_row = [text.alphabet[text.onehot_to_class(b).data[0] % text.alpha_len] for b in tensor]
    pos_row = [text.ix_to_pos[int(text.onehot_to_class(b).data[0] / text.alpha_len)] for b in tensor]
    if verbose is True:
        print ''.join(char_row)
        print '-'.join(pos_row)

    output = model(tensor)[-1]
    next=torch.multinomial(output.div(temp).exp(),1).data[0]
    char_row.append(text.alphabet[next%text.alpha_len])
    pos_row.append(text.ix_to_pos[int(next/text.alpha_len)])

    to_cat = Variable(torch.zeros(1,len(tensor[-1])))
    to_cat[0,next] = 1.
    tensor = torch.cat((tensor, to_cat),0)

    for i in range(gen_len-1):
        output = model([tensor[-1]])[-1]
        next = torch.multinomial(output.div(temp).exp(), 1).data[0]
        char_row.append(text.alphabet[next % text.alpha_len])
        pos_row.append(text.ix_to_pos[int(next / text.alpha_len)])
        to_cat = Variable(torch.zeros(1, len(tensor[-1])))
        to_cat[0, next] = 1.
        tensor = torch.cat((tensor, to_cat), 0)

    if verbose is True:
        print ''.join(char_row)
        print '-'.join(pos_row)
    return char_row, pos_row

def generate_pos_while(model, text, input_tensor, pos_type, stopping_char=' ', temp=1., gen_len=10, verbose = False):
    tensor = input_tensor
    char_row = [text.alphabet[text.onehot_to_class(b).data[0] % text.alpha_len] for b in tensor]
    pos_row = [text.ix_to_pos[int(text.onehot_to_class(b).data[0] / text.alpha_len)] for b in tensor]
    if verbose is True:
        print ''.join(char_row)
        print '-'.join(pos_row)

    output = model(tensor)[-1]
    next=torch.multinomial(output.div(temp).exp(),1).data[0]
    char_row.append(text.alphabet[next%text.alpha_len])
    pos_row.append(text.ix_to_pos[int(next/text.alpha_len)])

    to_cat = Variable(torch.zeros(1,len(tensor[-1])))
    to_cat[0,next] = 1.
    tensor = torch.cat((tensor, to_cat),0)
    #print output
    for i in range(gen_len-1):
        output = model([tensor[-1]])[-1]
        temp_out = output.data
        for ix, out in enumerate(temp_out):
            if (ix >= (pos_type)*text.alpha_len) and (ix < (pos_type+1)*text.alpha_len):
                temp_out[ix] = out
                #print text.alphabet[int(ix%text.alpha_len)]
            else:
                temp_out[ix] = -1000
        output = Variable(torch.Tensor(temp_out))
        for ix, prob, in enumerate(output.div(temp).exp()):
            if (prob.data[0] > 0) and (verbose is True):
                print text.ix_to_pos[int(ix / text.alpha_len)], prob.data[0]
        next = torch.multinomial(output.div(temp).exp(), 1).data[0]
        char_row.append(text.alphabet[next % text.alpha_len])
        pos_row.append(text.ix_to_pos[int(next / text.alpha_len)])
        to_cat = Variable(torch.zeros(1, len(tensor[-1])))
        to_cat[0, next] = 1.
        tensor = torch.cat((tensor, to_cat), 0)
        if char_row[-1] == stopping_char:
            break

    if verbose is True:
        print ''.join(char_row)
        print '-'.join(pos_row)
    return char_row, pos_row

def generate_many_pos_while(model, text, input_tensor, follow_text, follow_pos,
                            pos_type, stopping_char=' ', temp=1., gen_len=10, verbose = False):
    reps = 30
    look_back = 40

    look_forward = 40
    all_pos = []
    all_char = []
    all_loss = []
    for reps in range(reps):
        tensor = input_tensor
        char_row = [text.alphabet[text.onehot_to_class(b).data[0] % text.alpha_len] for b in tensor]
        pos_row = [text.ix_to_pos[int(text.onehot_to_class(b).data[0] / text.alpha_len)] for b in tensor]
        if verbose is True:
            print ''.join(char_row)
            print '-'.join(pos_row)

        output = model(tensor)[-1]
        next=torch.multinomial(output.div(temp).exp(),1).data[0]
        char_row.append(text.alphabet[next%text.alpha_len])
        pos_row.append(text.ix_to_pos[int(next/text.alpha_len)])

        to_cat = Variable(torch.zeros(1,len(tensor[-1])))
        to_cat[0,next] = 1.
        tensor = torch.cat((tensor, to_cat),0)
        #print output
        for i in range(gen_len-1):
            output = model([tensor[-1]])[-1]
            temp_out = output.data
            for ix, out in enumerate(temp_out):
                if (ix >= (pos_type)*text.alpha_len) and (ix < (pos_type+1)*text.alpha_len):
                    temp_out[ix] = out
                    #print text.alphabet[int(ix%text.alpha_len)]
                else:
                    temp_out[ix] = -1000
            output = Variable(torch.Tensor(temp_out))
            for ix, prob, in enumerate(output.div(temp).exp()):
                if (prob.data[0] > 0) and (verbose is True):
                    print text.ix_to_pos[int(ix / text.alpha_len)], prob.data[0]
            next = torch.multinomial(output.div(temp).exp(), 1).data[0]
            char_row.append(text.alphabet[next % text.alpha_len])
            pos_row.append(text.ix_to_pos[int(next / text.alpha_len)])
            to_cat = Variable(torch.zeros(1, len(tensor[-1])))
            to_cat[0, next] = 1.
            tensor = torch.cat((tensor, to_cat), 0)
            if char_row[-1] == stopping_char:
                break

        if verbose is True:
            print ''.join(char_row)
            print '-'.join(pos_row)
            print pos_row
        all_loss.append(np.mean([model.loss_func(model.forward(inp), tar) for inp, tar in
                               text.text_pos_to_batch(''.join(char_row[-look_back:])+follow_text[:look_forward],
                                                      [text.pos_to_ix[p] for p in pos_row[-look_back:]]+follow_pos[:look_forward],
                                                      stride=1)]
                              ).data[0])
        all_char.append(char_row)
        all_pos.append(pos_row)
    best = np.argmin(all_loss)
    print all_loss
    return all_char[best], all_pos[best]