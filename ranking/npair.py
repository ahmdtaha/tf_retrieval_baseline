import tensorflow as tf

def npairs_loss_helper(labels, embeddings_anchor, embeddings_positive,
                reg_lambda=0.002, print_losses=False):
  """Computes the npairs loss.
  Npairs loss expects paired data where a pair is composed of samples from the
  same labels and each pairs in the minibatch have different labels. The loss
  has two components. The first component is the L2 regularizer on the
  embedding vectors. The second component is the sum of cross entropy loss
  which takes each row of the pair-wise similarity matrix as logits and
  the remapped one-hot labels as labels.
  See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
  Args:
    labels: 1-D tf.int32 `Tensor` of shape [batch_size/2].
    embeddings_anchor: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
      embedding vectors for the anchor images. Embeddings should not be
      l2 normalized.
    embeddings_positive: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
      embedding vectors for the positive images. Embeddings should not be
      l2 normalized.
    reg_lambda: Float. L2 regularization term on the embedding vectors.
    print_losses: Boolean. Option to print the xent and l2loss.
  Returns:
    npairs_loss: tf.float32 scalar.
  """
  # pylint: enable=line-too-long
  # Add the regularizer on the embedding.
  reg_anchor = tf.reduce_mean(
      tf.reduce_sum(tf.square(embeddings_anchor), 1))
  reg_positive = tf.reduce_mean(
      tf.reduce_sum(tf.square(embeddings_positive), 1))
  l2loss = tf.multiply(
      0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

  # Get per pair similarities.
  similarity_matrix = tf.matmul(
      embeddings_anchor, embeddings_positive, transpose_a=False,
      transpose_b=True)

  # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
  lshape = tf.shape(labels)
  assert lshape.shape == 1
  labels = tf.reshape(labels, [lshape[0], 1])

  labels_remapped = tf.cast(
      tf.equal(labels, tf.transpose(labels)), tf.float32)
  labels_remapped /= tf.reduce_sum(labels_remapped, 1, keepdims=True)

  # Add the softmax loss.
  xent_loss = tf.nn.softmax_cross_entropy_with_logits(
      logits=similarity_matrix, labels=labels_remapped)
  xent_loss = tf.reduce_mean(xent_loss, name='xentropy')

  if print_losses:
    xent_loss = tf.Print(
        xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])

  return l2loss + xent_loss

def npairs_loss(labels,embeddings_anchor,embeddings_positive,reg_lambda=0.002,print_losses=False):
    return npairs_loss_helper(labels, embeddings_anchor,embeddings_positive,reg_lambda=reg_lambda,print_losses=print_losses)

    # return tf.contrib.losses.metric_learning.npairs_loss(labels, embeddings_anchor,embeddings_positive,reg_lambda=reg_lambda,print_losses=print_losses)
