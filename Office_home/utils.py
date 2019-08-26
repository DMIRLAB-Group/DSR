from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import pandas as pd


def load_resnet_office(source_name, target_name, data_folder):
    source_path = os.path.join(data_folder, source_name + "_" + source_name + ".csv")
    target_path = os.path.join(data_folder, source_name + "_" + target_name + ".csv")

    source_data = pd.read_csv(source_path, header=None).values
    source_feature = source_data[:, :-1]
    source_label = np.array(source_data[:, -1], dtype=np.int32)

    target_data = pd.read_csv(target_path, header=None).values
    target_feature = target_data[:, :-1]
    target_label = np.array(target_data[:, -1], dtype=np.int32)

    return source_feature, source_label, target_feature, target_label, target_feature, target_label


def weight_variable(name, shape, trainable=True):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
                           trainable=trainable)


def bias_variable(name, shape, trainable=True):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
                           trainable=trainable)


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator (data, batch_size, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]

