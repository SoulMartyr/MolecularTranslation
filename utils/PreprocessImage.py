import os

import cv2
import numpy as np
from imgaug import augmenters as iaa

image_size = 224


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


data_dir = r"D:\GraduationProject\MolecularTranslation\data\molecular-translation"
out_dir = r"D:\GraduationProject\MolecularTranslation\data\preprocess_data"

count = 0
for dir in os.listdir(data_dir):
    dir_path = os.path.join(data_dir, dir)
    out_path = os.path.join(out_dir, dir)
    if not os.path.isdir(dir_path):
        continue
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(count)
    for d1 in os.listdir(dir_path):
        dir_path1 = os.path.join(dir_path, d1)
        out_path1 = os.path.join(out_path, d1)
        if not os.path.exists(out_path1):
            os.makedirs(out_path1)
        for d2 in os.listdir(dir_path1):
            dir_path2 = os.path.join(dir_path1, d2)
            out_path2 = os.path.join(out_path1, d2)
            if not os.path.exists(out_path2):
                os.makedirs(out_path2)
            for d3 in os.listdir(dir_path2):
                dir_path3 = os.path.join(dir_path2, d3)
                out_path3 = os.path.join(out_path2, d3)
                if not os.path.exists(out_path3):
                    os.makedirs(out_path3)
                for img_name in os.listdir(dir_path3):
                    img_path = os.path.join(dir_path3, img_name)
                    out_path4 = os.path.join(out_path3, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if dir is 'train':
                        img = vague_augment().augment_image(img)
                    img = pad_and_resize_augment(img)
                    cv2.imwrite(out_path4, img)
                    count += 1
