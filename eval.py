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

    if not model_is_exist:
        return

    with util.StopWatch() as sw:
        # Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=lambda: tf_util.eval_input_fn(test_data, test_label, 1000))

    print('Test set accuracy: {0:0.3f} in {1:0.3f}sec'.format(
        eval_result['accuracy'], sw.elapsed))
    stat_log.put_eval_stat(batch_size, train_steps, hidden_units, sw.elapsed, eval_result['accuracy'])


def main(argv):
    main_multi()

def main_multi():

    for batch_size, train_step, hidden_units in util.test_iter():
        try:
            do_test(batch_size, train_step, hidden_units)

            stat_log.unlog_err(batch_size, train_step, hidden_units)
        except:
            stat_log.log_err(batch_size, train_step, hidden_units)
            import traceback
            traceback.print_exc()
            # 評価途中とかそんな感じだと思われる

def main_one():
    batch_size = 1000    # NOTE 変更して遊んでみるポイント(一回のstepで使うデータの量かな）
    train_steps = 1000   # NOTE 変更して遊んでみるポイント(学習する回数と理解）
    hidden_units=[256, 256, 256, 256],   # NOTE 変更して遊んでみるポイント
    do_test(batch_size, train_steps, hidden_units)


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
