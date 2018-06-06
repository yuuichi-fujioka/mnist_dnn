# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tf_util
import util
import stat_log

def do_test(batch_size, train_steps, hidden_units):

    test_name = "dnn-{0}-{1}-{2}".format(
            batch_size,
            train_steps,
            "_".join([str(n) for n in hidden_units]))
    print('###### {} ######'.format(test_name))


    (train_data, train_label), (test_data, test_label), feature_columns = tf_util.load_data()

    classifier, model_is_exist = tf_util.make_classifier(batch_size, train_steps, hidden_units, feature_columns)

    if model_is_exist:
        return

    with util.StopWatch() as sw:
        # Train the Model.
        classifier.train(
            input_fn=lambda:tf_util.train_input_fn(train_data, train_label, batch_size),
            steps=train_steps)

    print('Training {0:0.3f}sec'.format(sw.elapsed))
    stat_log.put_train_stat(batch_size, train_steps, hidden_units, sw.elapsed)

def main(argv):
    main_multi()

import multiprocessing

_POOL_SIZE = multiprocessing.cpu_count()

def w(args):
    return do_test(*args)

def main_multi():
    with multiprocessing.Pool(_POOL_SIZE) as p:
        p.map(w, util.test_iter())
        p.close()

def main_one():
    batch_size = 1000    # NOTE 変更して遊んでみるポイント(一回のstepで使うデータの量かな）
    train_steps = 1000   # NOTE 変更して遊んでみるポイント(学習する回数と理解）
    hidden_units=[256, 256, 256, 256],   # NOTE 変更して遊んでみるポイント
    do_test(batch_size, train_steps, hidden_units)

if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
