import re
import time

import torch
from torch.nn import Module
from torch.nn import functional as F


class HSigmoid(Module):
    def __init__(self):
        super(HSigmoid, self).__init__()

    def forward(self, x):
        return F.relu6(x + 3.) / 6.


class Swish(Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class HSwish(Module):
    def __init__(self):
        super(HSwish, self).__init__()

    def forward(self, x):
        return F.relu6(x + 3.) / 6. * x


def spilt_formula_layer(formula):
    string = ''
    for i in re.findall(r"[A-Z][^A-Z]*", formula):
        element = re.match(r"\D+", i).group()
        num = i.replace(element, "")
        if num == "":
            string += "{} ".format(element)
        else:
            string += "{} {} ".format(element, str(num))
    return string.rstrip(' ')


def spilt_other_layers(formula):
    result = ""
    for form in formula:
        temp_result = ''
        for i in re.findall(r"[a-z][^a-z]*", form):
            element = i[0]
            all_num = i.replace(element, "").replace('/', "")
            num_sign = ''
            for j in re.findall(r"[0-9]+[^0-9]*", all_num):
                num_list = list(re.findall(r'\d+', j))
                assert len(num_list) == 1, "Get Number More Than 1"
                num = num_list[0]
                if j == num:
                    num_sign += "{} ".format(num)
                else:
                    sign = j.replace(num, "")
                    num_sign += "{} {} ".format(num, " ".join(list(sign)))
            temp_result += "/{} {}".format(element, num_sign)
        result += " " + temp_result.rstrip()
    return result.rstrip()


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])

    assert len(lr) == 1, "Get Learning Rate More Than 1"
    lr = lr[0]

    return lr


def get_time():
    now = time.localtime()
    now_time = time.strftime("%H:%M:%S", now)
    return now_time


def print_hint(hint):
    print(get_time(), hint)


def print_msg_head(epoch_num, batch_size):
    print('epoch_num = {}\n'.format(epoch_num))
    print('batch_size = {}\n'.format(batch_size))
    print('|---------------Info----------------|------Train------|------Valid------|\n')
    print('| time       epoch    iter   lr     | loss     dist   | loss     dist   |\n')
    print('-------------------------------------------------------------------------\n')


def print_msg(epoch, iteration, lr, train_accuracy, valid_accuracy, save_iteration):
    if iteration % save_iteration == 0:
        sign = '*'
    else:
        sign = ' '
    valid_accuracy = ['' if elem is None else elem for elem in valid_accuracy]
    print("| {:<8}   {:<6}   {:<4}   {:<6} | {:<6}   {:<6} | {:<6}   {:<6} |".format(get_time(), str(epoch)[:6],
                                                                                     str(str(iteration) + sign)[:4],
                                                                                     str(lr)[:5],
                                                                                     str(train_accuracy[0])[:6],
                                                                                     str(train_accuracy[1])[:6],
                                                                                     str(valid_accuracy[0])[:6],
                                                                                     str(valid_accuracy[1])[:6]))


def print_flush():
    print('\r', end='', flush=True)


def print_epoch(_iteration, batch_loss, batch_dist):
    print(_iteration, "Batch Average Loss:", batch_loss, ", Average Dist:", batch_dist)
