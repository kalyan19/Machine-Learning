from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import ceil, floor
from math import pi
from PIL import Image
from random import shuffle

# Imports
import numpy as np
import tensorflow as tf
import cv2
import os
import shutil

POS_DIR = 'Data/pos'
NEG_DIR = 'Data/neg'
IMAGE_SIZE = 100
POS_TRAIN_DIR = 'Data/pos_train'
POS_VALID_DIR = 'Data/pos_valid'
POS_EVAL_DIR = 'Data/pos_eval'
NEG_TRAIN_DIR = 'Data/neg_train'
NEG_VALID_DIR = 'Data/neg_valid'
NEG_EVAL_DIR = 'Data/neg_eval'
POS_AUG_DIR = 'Data/pos_augmented'
NEG_AUG_DIR = 'Data/neg_augmented'
POS_FINAL_DIR = 'Data/pos_final_train'
NEG_FINAL_DIR = 'Data/neg_final_train'


"""
Sets up the directory for validation, training, and evaluation
"""
def set_up():
    pos_num = len(os.listdir(POS_DIR))
    neg_num = len(os.listdir(NEG_DIR))

    # 867, 2642
    # for pos
    # 250 - validation
    # 250 - evaluation
    # rest (367) - training
    # for neg
    # 250 - valid
    # 250 - evaluation
    # rest (2142) - training
    # expand training for pos and neg --> 5000:5000

    pos_imgs = os.listdir(POS_DIR)
    shuffle(pos_imgs)
    eval_imgs, valid_imgs, pos_train_imgs = pos_imgs[:250], pos_imgs[250:500], pos_imgs[500:]

    for img_path in eval_imgs:
        img = cv2.imread(os.path.join(POS_DIR, img_path))
        cv2.imwrite(os.path.join(POS_EVAL_DIR, img_path), img)

    for img_path in valid_imgs:
        img = cv2.imread(os.path.join(POS_DIR, img_path))
        cv2.imwrite(os.path.join(POS_VALID_DIR, img_path), img)

    for img_path in pos_train_imgs:
        img = cv2.imread(os.path.join(POS_DIR, img_path))
        cv2.imwrite(os.path.join(POS_TRAIN_DIR, img_path), img)


    neg_imgs = os.listdir(NEG_DIR)
    shuffle(neg_imgs)
    eval_imgs, valid_imgs, neg_train_imgs = neg_imgs[:250], neg_imgs[250:500], neg_imgs[500:]

    for img_path in eval_imgs:
        img = cv2.imread(os.path.join(NEG_DIR, img_path))
        cv2.imwrite(os.path.join(NEG_EVAL_DIR, img_path), img)

    for img_path in valid_imgs:
        img = cv2.imread(os.path.join(NEG_DIR, img_path))
        cv2.imwrite(os.path.join(NEG_VALID_DIR, img_path), img)

    for img_path in neg_train_imgs:
        img = cv2.imread(os.path.join(NEG_DIR, img_path))
        cv2.imwrite(os.path.join(NEG_TRAIN_DIR, img_path), img)

    # now augment the data to get 15x for pos and 3x for neg
    pos_aug_imgs = os.listdir(POS_AUG_DIR)
    shuffle(pos_aug_imgs)
    size = 10000
    #pos_img_paths = train_imgs + pos_aug_imgs[:size - len(pos_train_imgs)]


    src_files = os.listdir(POS_TRAIN_DIR)
    for file_name in src_files:
        if (file_name.endswith('.png')):
            full_file_name = os.path.join(POS_TRAIN_DIR, file_name)
            shutil.copy(full_file_name, POS_FINAL_DIR)

    for file_name in pos_aug_imgs[:(size - len(pos_train_imgs))]:
        if (file_name.endswith('.png')):
            full_file_name = os.path.join(POS_AUG_DIR, file_name)
            shutil.copy(full_file_name, POS_FINAL_DIR)

    neg_aug_imgs = os.listdir(NEG_AUG_DIR)
    shuffle(neg_aug_imgs)
    #neg_img_paths = train_imgs + pos_aug_imgs[:size - len(neg_train_imgs)]

    src_files = os.listdir(NEG_TRAIN_DIR)
    for file_name in src_files:
        if (file_name.endswith('.png')):
            full_file_name = os.path.join(NEG_TRAIN_DIR, file_name)
            shutil.copy(full_file_name, NEG_FINAL_DIR)

    for file_name in neg_aug_imgs[:(size - len(neg_train_imgs))]:
        if (file_name.endswith('.png')):
            full_file_name = os.path.join(NEG_AUG_DIR, file_name)
            shutil.copy(full_file_name, NEG_FINAL_DIR)

def printSize():
    pos_num = len(os.listdir(POS_FINAL_DIR))
    neg_num = len(os.listdir(NEG_FINAL_DIR))

    print(pos_num, neg_num)

def main():
    #set_up()
    #printSize()


main()
