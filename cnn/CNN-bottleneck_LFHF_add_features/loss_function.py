import tensorflow as tf
import numpy as np

class Loss:

    def __init__(self,true, pred):

        self.pred = pred
        self.true = true
        self.ce = self.Cross_entropy()

    def Cross_entropy(self):

        self.loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=self.true, logits=self.pred)

        return self.loss_op
