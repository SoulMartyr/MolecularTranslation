import pickle
import random
from collections import defaultdict

import Levenshtein
import cv2
import numpy as np
import pandas as pd
import torch
from imgaug import augmenters as iaa
from rdkit import Chem
from rdkit import RDLogger
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

from utils.Util import spilt_formula_layer, spilt_other_layers

RDLogger.DisableLog('rdApp.*')

tqdm.pandas(desc='apply')

data_dir = r'D:\GraduationProject\MolecularTranslation\data'
image_size = 224
vocab_size = 193
padding_value = 192
max_seq_length = 300


class Tokenizer(object):
    def __init__(self, is_load=True):
        self.word_dict = {}
        self.num_dict = {}

        if is_load:
            with open(data_dir + '/tokenizer_word_dict.pickle', 'rb') as f:
                self.word_dict = pickle.load(f)
            self.num_dict = {i: s for s, i in self.word_dict.items()}

    def build_vocab(self, text):
        vocab = set()
        for t in text:
            vocab.update(t.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.word_dict[s] = i
        self.num_dict = {i: s for s, i in self.word_dict.items()}

    def __len__(self):
        return len(self.word_dict)

    def text_to_sequence(self, text):
        sequence = list()
        sequence.append(self.word_dict['<sos>'])
        for word in text.split(' '):
            sequence.append(self.word_dict[word])
        sequence.append(self.word_dict['<eos>'])
        return sequence

    def texts_to_sequences(self, texts):
        sequences = [self.text_to_sequence(text) for text in texts]
        return sequences

    def sequence_to_text(self, sequence):
        text = str()
        for num in sequence:
            num = self.num_dict[num]
            if num == '<sos>':
                continue
            if num == '<eos>' or num == '<pad>':
                break
            text += num
        return text

    def sequences_to_texts(self, sequences):
        texts = [self.sequence_to_text(sequence) for sequence in sequences]
        return texts

    def predict(self, predict):
        formula = 'InChI=1S/'
        # 遍历预测
        for num in predict:
            # 遇到 <eos> 或 <pad> 中止
            if num == self.word_dict['<eos>'] or num == self.word_dict['<pad>']:
                break
            formula += self.num_dict[num]
        return formula

    def predicts(self, predicts):
        formulas = [
            self.predict(predict)
            for predict in predicts
        ]
        return formulas


def load_tokenizer():
    tokenizer = Tokenizer(is_load=True)
    # print("The Vocab Size is {}".format(len(tokenizer)))
    # print("Load Tokenizer Success.")
    return tokenizer


def padding(sequence, value, max_length):
    batch_size = len(sequence)
    padding_sequence = np.full((batch_size, max_length), fill_value=value, dtype=np.int32)
    for idx, seq in enumerate(sequence):
        seq_length = len(seq)
        padding_sequence[idx, :seq_length] = seq
    return padding_sequence


def vague_augment():
    return iaa.OneOf([iaa.Noop(), iaa.SaltAndPepper(0.001), iaa.SaltAndPepper(0.002)])


def pad_and_resize_augment(image):
    image = 255 - image

    image = image[0 < image.sum(axis=1), :]
    image = image[:, 0 < image.sum(axis=0)]

    h, w = image.shape
    max_len = max(h, w)
    out = np.zeros((max_len, max_len), dtype=np.uint8)

    center = max_len // 2
    center_x, center_y = w // 2, h // 2
    y = center - center_y
    x = center - center_x

    out[y:y + h, x:x + w] = image

    out = iaa.Resize((image_size, image_size)).augment_image(out)
    out = 255 - out
    return out


def rotate_augment(image, orientation):
    if orientation == 1:
        image = np.rot90(image, -1)
    elif orientation == 2:
        image = np.rot90(image, 1)
    elif orientation == 3:
        image = np.rot90(image, 2)
    return image


class MolecularDataset(Dataset):
    def __init__(self, data, tokenizer, data_path=data_dir + '/molecular-translation/', mode='train'):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.mode = mode
        self.data_path = data_path

    def __str__(self):
        string = ''
        string += '\t len = %d\n' % len(self)
        string += '\t data  = %s\n' % str(self.data.shape)
        return string

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_info = self.data.iloc[index]
        image_file = self.data_path + '{}'.format(self.mode) + '/{}/{}/{}/{}.png'.format(
            image_info.image_id[0], image_info.image_id[1], image_info.image_id[2], image_info.image_id)
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

        if self.mode == "test":
            image = rotate_augment(image, image_info.orientation)
        if self.mode == "train":
            image = vague_augment().augment_image(image)
        image = pad_and_resize_augment(image)

        info = {
            'index': index,
            'image_id': image_info.image_id,
            'image': image,
            'InChI': image_info.InChI,
            'formula': image_info.formula,
            'text': image_info.text,
            'sequence': image_info.sequence,
        }
        return info


class MolecularSampler(Sampler):
    def __init__(self, data_source, length=-1, is_shuffle=False):
        if not length > 0:
            length = len(data_source)
        self.data_source = data_source
        self.length = length
        self.is_shuffle = is_shuffle

    def __len__(self):
        return self.length

    def __iter__(self):
        index = np.arange(self.length)
        if self.is_shuffle:
            random.shuffle(index)
        return iter(index)


def collate_fn(batch, is_sort=True):
    collate = defaultdict(list)

    if is_sort:
        sorted_idx = np.argsort([-len(info['sequence']) for info in batch])
        batch = [batch[idx] for idx in sorted_idx]

    for info in batch:
        for key, value in info.items():
            collate[key].append(value)

    collate['length'] = [len(sequence) for sequence in collate['sequence']]

    sequence = [np.array(sequence, dtype=np.int32) for sequence in collate['sequence']]
    sequence = padding(sequence, max_length=max_seq_length, value=padding_value)
    collate['sequence'] = torch.from_numpy(sequence).long()

    image = np.stack(collate['image'])
    image = image.astype(np.float32) / 255
    collate['image'] = torch.from_numpy(image).unsqueeze(1).repeat(1, 3, 1, 1)

    return collate


def generate_train_data(fold=6):
    tokenizer = load_tokenizer()
    data = pd.read_csv(data_dir + '/molecular-translation/train_labels.csv')
    data['formula'] = data.InChI.progress_apply(lambda x: x.split('/')[1])
    data['text'] = data.formula.progress_apply(
        lambda x: spilt_formula_layer(x)) + data.InChI.progress_apply(
        lambda x: spilt_other_layers(x.split('/')[2:]))
    data['sequence'] = data.text.progress_apply(lambda x: tokenizer.text_to_sequence(x))
    data['length'] = data.sequence.progress_apply(lambda x: len(x) - 2)

    rd = random.random
    random.seed(427)
    fold_list = (len(data) // fold) * [i for i in range(fold)]
    random.shuffle(fold_list, random=rd)
    data['fold'] = fold_list

    data.to_csv(data_dir + '/train_data.csv')


def generate_test_data():
    data = pd.read_csv(data_dir + '/molecular-translation/sample_submission.csv')
    orientation = pd.read_csv(data_dir + '/test_orientation.csv')
    data = data.merge(orientation, on='image_id')

    data.loc[:, 'InChI'] = ""
    data.loc[:, 'formula'] = ""
    data.loc[:, 'text'] = ""
    data.loc[:, 'sequence'] = ""
    data.loc[:, 'length'] = 0

    data.to_csv(data_dir + '/test_data.csv')


def cross_entropy_loss_cuda(probability, truth, length):
    truth = truth[:, 1:]
    length_list = [l - 1 for l in length]
    probability = pack_padded_sequence(probability, length_list, batch_first=True).data
    truth = pack_padded_sequence(truth, length_list, batch_first=True).data
    loss = F.cross_entropy(probability, truth, ignore_index=192)
    return loss


def cross_entropy_loss_numpy(probability, truth):
    batch_size = len(probability)
    truth = truth.reshape(-1)
    probability = probability[np.arange(batch_size), truth]
    loss = -np.log(np.clip(probability, 1e-6, 1))
    loss = loss.mean()
    return loss


def calculate_edit_distance(predict, truth, tokenizer):
    score = []
    predict = tokenizer.sequences_to_texts(predict)
    truth = tokenizer.sequences_to_texts(truth)
    for p, t in zip(predict, truth):
        s = Levenshtein.distance(p, t)
        score.append(s)
    score = np.array(score)
    return score.mean()


def valid(valid_loader, net):
    valid_probability = []
    valid_truth = []
    valid_num = 0
    tokenizer = load_tokenizer()
    net.eval()
    for _, batch in enumerate(valid_loader):
        batch_size = len(batch['index'])
        image = batch['image'].cuda()
        sequence = batch['sequence'].cuda()
        length = batch['length']
        with torch.no_grad():
            result = net(image, sequence)
            probability = F.softmax(result, -1)

        valid_num += batch_size
        valid_probability.append(probability.data.cpu().numpy())
        valid_truth.append(sequence.data.cpu().numpy())

    assert (valid_num == len(valid_loader.sampler))

    probability = np.concatenate(valid_probability)
    predict = probability.argmax(-1)
    truth = np.concatenate(valid_truth)
    edit_distance = calculate_edit_distance(predict, truth, tokenizer)

    probability = probability[:, :-1].reshape(-1, vocab_size)
    truth = truth[:, 1:].reshape(-1)

    non_padding = np.where(truth != padding_value)[0]
    probability = probability[non_padding]
    truth = truth[non_padding]
    loss = cross_entropy_loss_numpy(probability, truth)

    return [loss, edit_distance]


def normalize_inchi(inchi):
    try:
        mol = Chem.MolFromInchi(inchi)
        return inchi if (mol is None) else Chem.MolToInchi(mol)
    except:
        return inchi


def predict(test_loader, net):
    tokenizer = load_tokenizer()
    result = []
    test_num = 0
    for t, batch in enumerate(test_loader):
        batch_size = len(batch['image'])
        image_batch = batch['image'].cuda()
        for idx in range(batch_size):
            image = image_batch[idx, :]
            net.eval()
            with torch.no_grad():
                out = net.forward_predict(image)
                out = out.data.cpu().numpy()
                out = tokenizer.predict(out)
                out = normalize_inchi(out)
                result.extend(out)

        test_num += batch_size

    assert (test_num == len(test_loader.dataset)), "Run Test Num Error!"
    return result
