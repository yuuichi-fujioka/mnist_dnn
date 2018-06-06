# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import plyvel
import prettytable

import util

_LOCK = multiprocessing.Lock()

_PATH = "./db/"


def lock(func):
    def w(*args, **argv):
        with _LOCK:
            return func(*args, **argv)
    return w

def _db():
    return plyvel.DB(_PATH, create_if_missing=True)


@lock
def put(k, v):
    db = _db()
    db.put(k.encode('utf-8'), v.encode('utf-8'))
    db.close()


@lock
def get(k):
    db = _db()
    v = db.get(k.encode('utf-8'))
    db.close()
    return v


@lock
def delete(k):
    db = _db()
    v = db.delete(k.encode('utf-8'))
    db.close()
    return v


def _db_key(batch_size, train_steps, hidden_units):
    return "{0}/{1}/{2}".format(
        batch_size,
        train_steps,
        "_".join([str(n) for n in hidden_units]))


def put_train_stat(batch_size, train_steps, hidden_units, time_sec):
    k = "train/{0}".format(_db_key(batch_size, train_steps, hidden_units))
    put(k, "{0:0.3f}".format(time_sec))


def put_prediction_stat(batch_size, train_steps, hidden_units, time_sec):
    k = "prediction/{0}".format(_db_key(batch_size, train_steps, hidden_units))
    put(k, "{0:0.3f}".format(time_sec))


def put_eval_stat(batch_size, train_steps, hidden_units, time_sec, accuracy):
    k = "eval/{0}".format(_db_key(batch_size, train_steps, hidden_units))
    put("/".join([k, "time"]), "{0:0.3f}".format(time_sec))
    put("/".join([k, "accuracy"]), "{0:0.3f}".format(accuracy))
    delete(k)


def log_err(batch_size, train_steps, hidden_units):
    k = "err/{0}".format(_db_key(batch_size, train_steps, hidden_units))
    put(k, 'error')


def unlog_err(batch_size, train_steps, hidden_units):
    k = "err/{0}".format(_db_key(batch_size, train_steps, hidden_units))
    delete(k)


@lock
def get_all():
    db = _db()
    values = []
    def _(k):
        v = db.get(k.encode('utf-8'))
        if v:
            return str(v, encoding='utf-8')
        else:
            return ""

    for b, t, h in util.test_iter():
        k = _db_key(b, t, h)
        train_time = _("/".join(["train", k]))
        eval_time =  _("/".join(["eval", k, "time"]))
        eval_accu =  _("/".join(["eval", k, "accuracy"]))
        pred_time =  _("/".join(["prediction", k]))
        err =        _("/".join(["err", k]))
        values.append((
            str(b), str(t), "_".join([str(n) for n in h]),
            train_time,
            eval_time,
            eval_accu,
            pred_time,
            err))

    return values


if __name__ == '__main__':
    columns = (
        ("BatchSize", "r"),
        ("TrainStep", "r"),
        ("Hidden Units", "l"),
        ("Train Time(sec)", "r"),
        ("Eval Time(sec)", "r"),
        ("Model Accuracy(%)", "r"),
        ("Prediction Time(microsec)", "r"),
        ("Has Error", "l"),
    )
    pt = prettytable.PrettyTable([c for c, _ in columns])
    for c, a in columns:
        pt.align[c] = a
    pt.sortby = "Model Accuracy(%)"
    pt.reversesort = True
    for v in get_all():
        pt.add_row(v)
    print(pt)
