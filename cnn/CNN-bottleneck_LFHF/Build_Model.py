import tensorflow as tf


class CNN:
    def __init__(self, name, prediction_output):
        self.name = name
        self.prediction_output = prediction_output

    def __call__(self, input):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            return self._build_model(input)

    def _build_model(self, input):
        print(input)

        conv11 = tf.layers.conv3d(inputs=input, filters=64, kernel_size=(3, 3, 3), padding='valid',
                                  data_format="channels_first",
                                  activation=tf.nn.relu, name=self.name + "_conv_11")
        print(conv11)
        pool1 = tf.layers.max_pooling3d(inputs=conv11, pool_size=(1, 3, 1), strides=(1, 3, 1),
                                        data_format="channels_first", name=self.name + "_mpool_1")
        print(pool1)

        conv21 = tf.layers.conv3d(inputs=pool1, filters=128, kernel_size=(3, 3, 3), padding='valid',
                                  data_format="channels_first",
                                  activation=tf.nn.relu, name=self.name + "_conv_21")
        print(conv21)
        pool2 = tf.layers.max_pooling3d(inputs=conv21, pool_size=(1, 3, 1), strides=(1, 3, 1),
                                        data_format="channels_first", name=self.name + "_mpool_2")
        print(pool2)

        conv31 = tf.layers.conv3d(inputs=pool2, filters=256, kernel_size=(3, 3, 3), padding='valid',
                                  data_format="channels_first",
                                  activation=tf.nn.relu, name=self.name + "_conv_31")
        print(conv31)
        pool3 = tf.layers.max_pooling3d(inputs=conv31, pool_size=(3, 3, 3), strides=(3, 3, 3),
                                        data_format="channels_first", name=self.name + "_mpool_3")
        print(pool3)

        conv41 = tf.layers.conv3d(inputs=pool3, filters=256, kernel_size=(3, 3, 3), padding='valid',
                                  data_format="channels_first",
                                  activation=tf.nn.relu, name=self.name + "_conv_41")
        print(conv41)
        pool4 = tf.layers.max_pooling3d(inputs=conv41, pool_size=(3, 3, 3), strides=(3, 3, 3),
                                        data_format="channels_first", name=self.name + "_mpool_4")
        print(pool4)

        conv51 = tf.layers.conv3d(inputs=pool4, filters=128, kernel_size=(1, 1, 1), padding='valid',
                                  data_format="channels_first",
                                  activation=tf.nn.relu, name=self.name + "_conv_51")
        print(conv51)
        pool5 = tf.layers.max_pooling3d(inputs=conv51, pool_size=(3, 3, 3), strides=(3, 3, 3),
                                        data_format="channels_first", name=self.name + "_mpool_5")
        print(pool5)

        flattened = tf.layers.flatten(pool5, name=self.name + "_flatten")
        print(flattened)

        dp1 = tf.layers.dropout(flattened, 0.7, name=self.name + "_dp_1")
        print(dp1)

        fc1 = tf.layers.dense(inputs=dp1, units=64, activation=tf.nn.relu, name=self.name + "_fc_1")
        print(fc1)

        fc2 = tf.layers.dense(inputs=fc1, units=16, activation=None, name=self.name + "_fc_2")
        print(fc2)

        fc3 = tf.layers.dense(inputs=fc2, units=64, activation=tf.nn.relu, name=self.name + "_fc_3")
        print(fc3)

        out = tf.layers.dense(inputs=fc3, units=self.prediction_output, activation=tf.nn.softmax, name=self.name + "_output")
        print(out)

        return out

