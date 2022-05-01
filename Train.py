import os

import numpy as np
import pandas as pd
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

import About
from net.Net import AmpNet
from net.optim.Lookahead import Lookahead
from net.optim.RAdam import RAdam
from utils import Util, DataUtil

version = About.version
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
out_dir = '../checkpoint/v{}'.format(version)

image_size = 224
vocab_size = 193
max_length = 300

learning_rate = 0.0001
batch_size = 32
epoch_num = 80000
log_iteration = 1000
valid_iteration = 5000
save_iteration = 5000

train_loss = torch.FloatTensor([0]).cuda().sum()
valid_loss = torch.FloatTensor([0]).cuda().sum()
train_accuracy = np.zeros(2, np.float32)
valid_accuracy = np.zeros(2, np.float32)
epoch_accuracy = np.zeros(2, np.float32)

train_data = pd.read_pickle('./data/train_data.pkl')
valid_data = pd.read_pickle('./data/valid_data.pkl')

tokenizer = DataUtil.load_tokenizer()
train_dataset = DataUtil.MolecularDataset(train_data, tokenizer)
valid_dataset = DataUtil.MolecularDataset(valid_data, tokenizer)

train_loader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=DataUtil.collate_fn,
)
valid_loader = DataLoader(
    valid_dataset,
    sampler=DataUtil.MolecularSampler(valid_dataset, 5000),
    batch_size=32,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=DataUtil.collate_fn,
)

Util.print_hint("Load DataSet Success")

grad_scaler = amp.GradScaler()
net = AmpNet().cuda()
weight_path = None
if weight_path is not None:
    weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
    start_iteration = weight['iteration']
    start_epoch = weight['epoch']
    state_dict = weight['state_dict']
    net.load_state_dict(state_dict, strict=False)
else:
    start_iteration = 0
    start_epoch = 0

optimizer = Lookahead(RAdam(filter(lambda x: x.requires_grad, net.parameters()), lr=learning_rate), alpha=0.5, k=5)
Util.print_hint("Load Net Success")

Util.print_msg_head(epoch_num, batch_size)
iteration = start_iteration
epoch = start_epoch
rate = 0
while epoch < epoch_num:
    for _, batch in enumerate(train_loader):

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        if iteration % save_iteration == 0 and iteration != start_iteration:
            torch.save({
                'iteration': iteration,
                'epoch': epoch,
                'state_dict': net.state_dict(),
            }, out_dir + '/weight/{}iter.pth'.format(iteration))
        valid_accuracy = [None, None]
        if iteration % valid_iteration == 0 and iteration != start_iteration:
            valid_accuracy = DataUtil.valid(valid_loader, net)

        batch_size = len(batch['index'])
        image = batch['image'].cuda()
        sequence = batch['sequence'].cuda()
        length = batch['length']
        net.train()
        optimizer.zero_grad()

        with amp.autocast():
            out = net(image, sequence)
            train_loss = DataUtil.cross_entropy_loss_cuda(out, sequence, length)
            sequence = sequence.detach().cpu().numpy()
            predict = out.detach().cpu().numpy().argmax(-1)
            train_accuracy = np.array(
                [train_loss.item(), DataUtil.calculate_edit_distance(predict, sequence, tokenizer)])

        if iteration % log_iteration == 0 and iteration != start_iteration:
            learning_rate = Util.get_learning_rate(optimizer)
            Util.print_flush()
            Util.print_msg(epoch, iteration, learning_rate, train_accuracy, valid_accuracy, save_iteration)

        epoch_accuracy += train_accuracy

        grad_scaler.scale(train_loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

    if iteration % log_iteration == 0:
        Util.print_epoch(log_iteration, epoch_accuracy[0] / 100, epoch_accuracy[1] / 100)
        epoch_accuracy[...] = 0
    iteration += 1
epoch += 1
Util.print_flush()
