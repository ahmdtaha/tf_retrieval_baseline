import tensorflow as tf
from tensorflow.contrib import slim

def head(intput, num_classes):

    output = slim.fully_connected(
        intput, num_classes, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer())

    return output
