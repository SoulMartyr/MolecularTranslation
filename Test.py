import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

import About
from net.Net import AmpNet
from utils.DataUtil import predict, MolecularDataset, load_tokenizer, collate_fn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
net = AmpNet()

version = About.version
out_dir = './result/v{}'.format(version)
net_checkpoint_dir = './checkpoint/v{}'.format(version)

test_data = pd.read_pickle('./data/test_data.pkl')
tokenizer = load_tokenizer()
test_dataset = MolecularDataset(test_data, tokenizer, mode='test')
test_loader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=16,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=lambda batch: collate_fn(batch, False)
)

net = AmpNet().cuda()
net.load_state_dict(torch.load(net_checkpoint_dir)['state_dict'])

predict_result = predict(test_data, net)

result_data = pd.DataFrame()
result_data.loc[:, 'image_id'] = test_data.image_id.values
result_data.loc[:, 'InChI'] = predict_result
result_data.to_csv(out_dir + '/submit.csv', index=False)
