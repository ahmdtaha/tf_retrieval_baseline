import tensorflow as tf

def lifted_loss(labels,embeddings,margin):
    return tf.contrib.losses.metric_learning.lifted_struct_loss(labels,embeddings,margin=margin)
