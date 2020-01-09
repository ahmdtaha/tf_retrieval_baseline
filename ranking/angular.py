import tensorflow as tf
import numpy as np

def angular_loss(input_labels, anchor_features, pos_features, degree=45, batch_size=10, with_l2reg=False):
    '''
    #NOTE: degree is degree!!! not radian value
    '''
    if with_l2reg:
        reg_anchor = tf.reduce_mean(tf.reduce_sum(tf.square(anchor_features), 1))
        reg_positive = tf.reduce_mean(tf.reduce_sum(tf.square(pos_features), 1))
        l2loss = tf.multiply(0.25 * 0.002, reg_anchor + reg_positive, name='l2loss_angular')
    else:
        l2loss = 0.0

    alpha = np.deg2rad(degree)
    sq_tan_alpha = np.tan(alpha) ** 2

    # anchor_features = tf.nn.l2_normalize(anchor_features)
    # pos_features = tf.nn.l2_normalize(pos_features)

    # 2(1+(tan(alpha))^2 * xaTxp)
    # batch_size = 10
    xaTxp = tf.matmul(anchor_features, pos_features, transpose_a=False, transpose_b=True)
    sim_matrix_1 = tf.multiply(2.0 * (1.0 + sq_tan_alpha) * xaTxp, tf.eye(batch_size, dtype=tf.float32))

    # 4((tan(alpha))^2(xa + xp)Txn
    xaPxpTxn = tf.matmul((anchor_features + pos_features), pos_features, transpose_a=False, transpose_b=True)
    sim_matrix_2 = tf.multiply(4.0 * sq_tan_alpha * xaPxpTxn,
                               tf.ones_like(xaPxpTxn, dtype=tf.float32) - tf.eye(batch_size, dtype=tf.float32))

    # similarity_matrix
    similarity_matrix = sim_matrix_1 + sim_matrix_2

    # do softmax cross-entropy
    lshape = tf.shape(input_labels)
    # assert lshape.shape == 1
    labels = tf.reshape(input_labels, [lshape[0], 1])

    labels_remapped = tf.cast(tf.equal(labels, tf.transpose(labels)),tf.float32)
    labels_remapped /= tf.reduce_sum(labels_remapped, 1, keepdims=True)

    xent_loss = tf.nn.softmax_cross_entropy_with_logits(logits=similarity_matrix, labels=labels_remapped)
    xent_loss = tf.reduce_mean(xent_loss, name='xentropy_angular')


    return l2loss + xent_loss