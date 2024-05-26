#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:10:22 2024

@author: mac
"""

import tensorflow as tf
import numpy as np
import random

SEED = 42

def set_seed(seed=SEED):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
