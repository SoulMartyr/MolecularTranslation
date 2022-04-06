from math import trunc, ceil
from random import choice

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def convert_to_path(split: str):
    def aux(image_id: str) -> str:
        return r"D:\GraduationProject\MolecularTranslation\data\molecular-translation\{}\{}\{}\{}\{}.png".format(
            split, image_id[0], image_id[1], image_id[2], image_id
        )

    return aux


train = pd.read_csv(r'D:\GraduationProject\MolecularTranslation\data\molecular-translation\train_labels.csv')

train_paths = train.image_id.apply(convert_to_path('train'))


class IMG_LOADER:

    def __init__(self, size=560, denoise=False):
        self.size = size
        self.denoise = denoise

        self.conv = nn.Conv2d(1, 1, (3, 3), padding=1, bias=False)
        self.conv.weight = torch.nn.Parameter(torch.ones(3, 3)[None, None, :, :])
        self.conv.require_grad = False

    def __call__(self, path):
        center = self.size // 2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # invert
        img = 255 - img

        # denoise
        if self.denoise:
            denoised = img.reshape(-1).copy()
            denoised[(self.conv(torch.Tensor(img[None, None, :, :])).detach().numpy() <= 255).squeeze().reshape(-1)] = 0
            img = denoised.reshape(img.shape)

        # smart crop
        img = img[0 < img.sum(axis=1), :]
        img = img[:, 0 < img.sum(axis=0)]

        # get size
        h, w = img.shape

        # longer edge
        scale = self.size / max(h, w)

        # resize
        img = cv2.resize(img, (trunc(scale * w), trunc(scale * h)))

        # padding
        h, w = img.shape
        center_x, center_y = ceil(w / 2), ceil(h / 2)

        _out = np.zeros((self.size, self.size), dtype=np.uint8)
        y = center - center_y
        x = center - center_x
        _out[y:y + h, x:x + w] = img

        return _out


img_size = 224
get_img = IMG_LOADER(img_size, False)
get_img_denoised = IMG_LOADER(img_size, True)

nrows, ncols = 5, 2

fig, ax = plt.subplots(nrows, ncols, figsize=(4, 7), dpi=400)

for i in range(5):
    p = choice(train_paths)
    img = get_img(p)
    cv2.imwrite('r_org.jpg', img)
    img = get_img_denoised(p)
    cv2.imwrite('r_denoise.jpg', img)
