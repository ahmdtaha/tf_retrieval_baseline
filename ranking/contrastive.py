import tensorflow as tf
from tensorflow.python.ops import math_ops

def contrastive_loss(labels, embeddings_anchor, embeddings_positive,
                     margin=1.0):
    """Computes the contrastive loss.
    This loss encourages the embedding to be close to each other for
      the samples of the same label and the embedding to be far apart at least
      by the margin constant for the samples of different labels.
    See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Args:
      labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
        binary labels indicating positive vs negative pair.
      embeddings_anchor: 2-D float `Tensor` of embedding vectors for the anchor
        images. Embeddings should be l2 normalized.
      embeddings_positive: 2-D float `Tensor` of embedding vectors for the
        positive images. Embeddings should be l2 normalized.
      margin: margin term in the loss definition.
    Returns:
      contrastive_loss: tf.float32 scalar.
    """
    # embeddings_anchor = tf.Print(embeddings_anchor,[tf.shape(embeddings_anchor),tf.shape(embeddings_positive)],'embeddings_anchor shapes')
    epsilon= 10e-6
    distances = math_ops.sqrt(
        math_ops.reduce_sum(
            math_ops.square(embeddings_anchor - embeddings_positive), 1) + epsilon)
    # distances = tf.Print(distances,[tf.shape(distances),distances],'distances ',summarize=1000)
    # Add contrastive loss for the siamese network.
    #   label here is {0,1} for neg, pos.

    pos_loss = math_ops.to_float(labels) * math_ops.square(distances)
    # pos_loss = tf.Print(pos_loss, [tf.shape(pos_loss),pos_loss], 'pos_loss ',summarize=1000)
    neg_loss = (1. - math_ops.to_float(labels)) * math_ops.square(math_ops.maximum(margin - distances, 0.))
    # neg_loss = tf.Print(neg_loss, [tf.shape(neg_loss),(1. - math_ops.to_float(labels)),math_ops.square(math_ops.maximum(margin - distances, 0.)),neg_loss], 'neg_loss ',summarize=1000)

    contrastive_loss = math_ops.reduce_mean(pos_loss + neg_loss, name='contrastive_loss')
    return contrastive_loss