# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import itertools

def test_iter():
    batch_sizes = [100, 500, 1000]
    train_steps = [100, 1000, 2000, 5000, 10000]
    hidden_units_list = [
            [256, 32],
            [32, 256],
            [32, 32],
            [256, 256],
            [32, 32, 32],
            [32, 32, 32, 32],
    ]
    return itertools.product(batch_sizes, train_steps, hidden_units_list)

import time

class StopWatch(object):
    def __enter__(self):
        self._start_at = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.time() - self._start_at
