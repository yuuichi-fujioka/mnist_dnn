# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import pandas


def make_classifier(batch_size, train_steps, hidden_units, feature_columns):
    model_dir = "./models/dnn/{0}/{1}/{2}".format(
            batch_size,
            train_steps,
            "_".join([str(n) for n in hidden_units]))

    model_is_exist = os.path.exists(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        optimizer=tf.train.AdamOptimizer(1e-4),
         dropout=0.1,
        n_classes=10,
        model_dir=model_dir)

    return classifier, model_is_exist


def load_mnist_data():
    # iris_dataと同じ形にしてみる

    # mnistの手書き文字学習用データ（d. = 学習用、 t. = 評価用、.x = 文字の画像、.y = ラベル)
    (dx, dy), (tx, ty) = tf.keras.datasets.mnist.load_data()
    dx = pandas.DataFrame(dx.reshape([60000, 28*28]), columns=[str(n) for n in range(28*28)])
    dy = pandas.Series(dy.astype('int32'))
    tx = pandas.DataFrame(tx.reshape([10000, 28*28]), columns=[str(n) for n in range(28*28)])
    ty = pandas.Series(ty.astype('int32'))

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in dx.keys():
        # NOTE 列の型と名前
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    return (dx, dy), (tx, ty), my_feature_columns


def load_data():
    return load_mnist_data()


def train_input_fn(features, labels, batch_size):
    # NOTE サンプルまま
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    # NOTE サンプルまま
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset
