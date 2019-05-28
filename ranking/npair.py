import tensorflow as tf

def npairs_loss(labels,embeddings_anchor,embeddings_positive,reg_lambda=0.002,print_losses=False):
    return tf.contrib.losses.metric_learning.npairs_loss(labels, embeddings_anchor,embeddings_positive,reg_lambda=reg_lambda,print_losses=print_losses)
