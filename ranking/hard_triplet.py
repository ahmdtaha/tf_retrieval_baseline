import numbers
import tensorflow as tf

def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.

    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).

    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        For convenience, if either `a` or `b` is a `Distribution` object, its
        mean is used.
    """
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)

def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.

    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """
    with tf.name_scope("cdist"):
        diffs = all_diffs(a, b)
        if metric == 'sqeuclidean':
            return tf.reduce_sum(tf.square(diffs), axis=-1)
        elif metric == 'euclidean':
            return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)
        elif metric == 'cityblock':
            return tf.reduce_sum(tf.abs(diffs), axis=-1)
        elif metric == 'cosine':
            # https://stackoverflow.com/questions/48485373/pairwise-cosine-similarity-using-tensorflow
            # normalized_input = tf.nn.l2_normalize(a, dim=1)
            # Embedding are assumed to be normalized
            prod = tf.matmul(a, b,adjoint_b=True)  # transpose second matrix
            return 1 - prod
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.format(metric))

def batch_hard(embeddings, pids, margin,metric):
    """Computes the batch-hard loss from arxiv.org/abs/1703.07737.

    Args:
        dists (2D tensor): A square all-to-all distance matrix as given by cdist.
        pids (1D tensor): The identities of the entries in `batch`, shape (B,).
            This can be of any type that can be compared, thus also a string.
        margin: The value of the margin if a number, alternatively the string
            'soft' for using the soft-margin formulation, or `None` for not
            using a margin at all.

    Returns:
        A 1D tensor of shape (B,) containing the loss value for each sample.
    """
    with tf.name_scope("batch_hard"):
        dists = cdist(embeddings, embeddings, metric=metric)

        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                      tf.expand_dims(pids, axis=0))
        # print(pids)
        # dists = tf.Print(dists, [dists], "Pair Dist", summarize=1000000)
        # same_identity_mask = tf.Print(same_identity_mask,[same_identity_mask, pids],"Hello World" ,summarize=1000000)
        negative_mask = tf.logical_not(same_identity_mask)
        positive_mask = tf.logical_xor(same_identity_mask,
                                       tf.eye(tf.shape(pids)[0], dtype=tf.bool))

        furthest_dist = dists*tf.cast(positive_mask, tf.float32)
        furthest_positive = tf.reduce_max(furthest_dist, axis=1)
        closest_negative = tf.map_fn(lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])),
                                     (dists, negative_mask), tf.float32)



        diff = (furthest_positive - closest_negative)
        diff = tf.squeeze(diff)
        #print(prefix,diff)
        # negative_idx = pids[negative_idx]
        if isinstance(margin, numbers.Real):
            diff_result = tf.maximum(diff + margin, 0.0)
            assert_op = tf.Assert(tf.equal(tf.rank(diff), 1), ['Rank of image must be equal to 1.'])
            with tf.control_dependencies([assert_op]):
                diff  = diff_result
        elif margin == 'soft':
            diff_result = tf.nn.softplus(diff)
            assert_op = tf.Assert(tf.equal(tf.rank(diff), 1), ['Rank of image must be equal to 1.'])
            with tf.control_dependencies([assert_op]):
                diff = diff_result
        elif margin.lower() == 'none':
            pass
        else:
            raise NotImplementedError(
                'The margin {} is not implemented in batch_hard'.format(margin))
        return diff
