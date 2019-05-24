import tensorflow as tf
from tensorflow.contrib import slim

def head(endpoints, embedding_dim, is_training, weights_regularizer=None):
    predict_var = 0
    input = endpoints['model_output']
    endpoints['head_output'] = slim.fully_connected(
        input, 1024, normalizer_fn=slim.batch_norm,
        normalizer_params={
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        },
        weights_regularizer=weights_regularizer
    )

    input_1 = endpoints['head_output']

    endpoints['emb_raw'] = slim.fully_connected(
        input_1, embedding_dim + predict_var, activation_fn=None,weights_regularizer=weights_regularizer,
        weights_initializer=tf.orthogonal_initializer(), scope='emb')


    endpoints['emb'] = tf.nn.l2_normalize(endpoints['emb_raw'], -1)
    # endpoints['data_sigma'] = None
    print('Normalize batch embedding')
    return endpoints
