import tensorflow as tf
import numpy as np

class opt:

    def __init__(self, learning_rate):

        self.learning_rate = learning_rate
        self.Adam = self.adam()

    def adam(self):

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        return self.optimizer
