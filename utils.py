import torch
from torch.autograd import Variable

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