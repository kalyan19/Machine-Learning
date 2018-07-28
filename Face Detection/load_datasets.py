from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import os
import cv2
import time


POS_VALID_DIR = 'Data/pos_valid'
POS_EVAL_DIR = 'Data/pos_eval'
NEG_VALID_DIR = 'Data/neg_valid'
NEG_EVAL_DIR = 'Data/neg_eval'
POS_FINAL_DIR = 'Data/pos_final_train'
NEG_FINAL_DIR = 'Data/neg_final_train'
IMAGE_SIZE = 128

def load_datasets():
    pos_file_names = ["{}/{}".format(POS_FINAL_DIR, file_name) for file_name in os.listdir(POS_FINAL_DIR) if file_name.endswith('.png')]
    pos_labels = np.full( len(pos_file_names), 1)
    neg_file_names = ["{}/{}".format(NEG_FINAL_DIR, file_name) for file_name in os.listdir(NEG_FINAL_DIR) if file_name.endswith('.png')]
    neg_labels = np.full( len(neg_file_names), 0)
    file_names =  np.concatenate((pos_file_names, neg_file_names), axis = 0)
    labels = np.concatenate((pos_labels, neg_labels), axis = 0)
    #train_data = np.asarray([convert_to_img(filename) for filename in file_names], dtype=np.float32)
    train_labels = labels
    train_dataset = file_names, train_labels

    pos_file_names = ["{}/{}".format(POS_VALID_DIR, file_name) for file_name in os.listdir(POS_VALID_DIR) if file_name.endswith('.png')]
    pos_labels = np.full( len(pos_file_names), 1)
    neg_file_names = ["{}/{}".format(NEG_VALID_DIR, file_name) for file_name in os.listdir(NEG_VALID_DIR) if file_name.endswith('.png')]
    neg_labels = np.full( len(neg_file_names), 0)
    file_names =  np.concatenate((pos_file_names, neg_file_names), axis = 0)
    labels = np.concatenate((pos_labels, neg_labels), axis = 0)
    #valid_data = np.asarray([convert_to_img(filename) for filename in file_names], dtype=np.float32)
    valid_labels = labels
    valid_dataset = file_names, valid_labels

    pos_file_names = ["{}/{}".format(POS_EVAL_DIR, file_name) for file_name in os.listdir(POS_EVAL_DIR) if file_name.endswith('.png')]
    pos_labels = np.full( len(pos_file_names), 1)
    neg_file_names = ["{}/{}".format(NEG_EVAL_DIR, file_name) for file_name in os.listdir(NEG_EVAL_DIR) if file_name.endswith('.png')]
    neg_labels = np.full( len(neg_file_names), 0)
    file_names =  np.concatenate((pos_file_names, neg_file_names), axis = 0)
    labels = np.concatenate((pos_labels, neg_labels), axis = 0)
    eval_data = np.asarray([convert_to_img(filename) for filename in file_names], dtype=np.float32)
    eval_labels = labels
    eval_dataset = eval_data, eval_labels

    return train_dataset, valid_dataset, eval_dataset


def convert_to_img(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    resized_image = resized_image.reshape((IMAGE_SIZE, IMAGE_SIZE, 1))
    return resized_image
