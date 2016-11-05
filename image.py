# -*- coding: utf-8 -*-
import os

import numpy as np
import cv2

IMAGE_SIZE = 64
#IMAGE_CHANNELS = 3
IMAGE_CHANNELS = 1


def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):

    def get_padding_size(image):
        # カラーとグレースケールで場合分け
        if len(image.shape) == 3:
            h, w, _ = image.shape
        else:
            h, w = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image


images = []
labels = []
def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        print(abs_path)
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path)
        else:                        # file
            if file_or_dir.endswith('.jpg') or file_or_dir.endswith('.JPG') or file_or_dir.endswith('.png'):
                image = read_image(abs_path)
                images.append(image)
                labels.append(path)

    return images, labels


def read_image(file_path):
    image = cv2.imread(file_path)
    if IMAGE_CHANNELS == 1:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return image


def extract_data(path):
    images, labels = traverse_dir(path)

    images = np.array(images)
    # ディレクトリ名を label として扱う
    labels = np.array([int(os.path.basename(label)) for label in labels])

    return images, labels
