# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

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

    if not model_is_exist:
        return

    test_data = test_data.drop(test_data.index[range(1,10000)])
    test_label = test_label.drop(test_label.index[range(1,10000)])

    with util.StopWatch() as sw:
        # predict the model.
        result = classifier.predict(
            input_fn=lambda: tf_util.eval_input_fn(test_data, None, 1))

    for pred_dict in result:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Predict: answer: {0}, predict: {1} {2:0.3f}% in {3:0.3f}micro sec'.format(
            test_label[0], class_id, probability * 100, sw.elapsed * 1000 * 1000))
        stat_log.put_prediction_stat(batch_size, train_steps, hidden_units, sw.elapsed * 1000 * 1000)


def main(argv):
    main_multi()

def main_multi():
    for batch_size, train_step, hidden_units in util.test_iter():
        try:
            do_test(batch_size, train_step, hidden_units)

            stat_log.unlog_err(batch_size, train_step, hidden_units)
        except:
            stat_log.log_err(batch_size, train_step, hidden_units)
            # 評価途中とかそんな感じだと思われる

def main_one():
    batch_size = 1000    # NOTE 変更して遊んでみるポイント(一回のstepで使うデータの量かな）
    train_steps = 1000   # NOTE 変更して遊んでみるポイント(学習する回数と理解）
    hidden_units=[256, 256, 256, 256],   # NOTE 変更して遊んでみるポイント
    do_test(batch_size, train_steps, hidden_units)


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
