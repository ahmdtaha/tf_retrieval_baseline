import tensorflow as tf

class FC1024Head(tf.keras.Model):
    def __init__(self, cfg):
        super(FC1024Head, self).__init__()
        self.h_1024 = tf.keras.layers.Dense(1025, activation=None,
                                          kernel_initializer=tf.keras.initializers.Orthogonal())
        self.batch_norm = tf.keras.layers.BatchNormalization(
            momentum = 0.9,
            epsilon=1e-5,
            scale=True,
        )
        self.head = tf.keras.layers.Dense(cfg.embedding_dim, activation=None,
                                                    kernel_initializer=tf.keras.initializers.Orthogonal())
    def call(self, inputs):
        h1 = tf.keras.backend.relu(self.batch_norm(self.h_1024(inputs)))
        return self.head(h1)


class DirectHead(tf.keras.Model):
    def __init__(self, cfg):
        super(DirectHead, self).__init__()
        self.head = tf.keras.layers.Dense(cfg.embedding_dim, activation=None,
                                                    kernel_initializer=tf.keras.initializers.Orthogonal())
    def call(self, inputs):
        return self.head(inputs)

class EmbeddingModel(tf.keras.Model):

    def __init__(self, cfg):
        super(EmbeddingModel, self).__init__()
        self.cfg = cfg

        if cfg.model_name == 'inception_v1':
            self.base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False)
            self.preprocess_input = tf.keras.applications.xception.preprocess_input
        elif cfg.model_name == 'resnet_v1_50':
            self.base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
            self.preprocess_input = tf.keras.applications.resnet.preprocess_input
        elif cfg.model_name == 'densenet169':
            self.base_model = tf.keras.applications.DenseNet169(weights='imagenet', include_top=False)
            self.preprocess_input = tf.keras.applications.densenet.preprocess_input
        else:
            raise NotImplementedError('Invalid model_name {}'.format(cfg.model_name))



        self.spatial_pooling = tf.keras.layers.GlobalAvgPool2D()
        if 'direct' in cfg.head_name:
            self.embedding_head = DirectHead(cfg)
        elif 'fc1024' in cfg.head_name:
            self.embedding_head = FC1024Head(cfg)
        else:
            raise NotImplementedError('Invalid head_name {}'.format(cfg.head_name))


        self.l2_embedding = 'normalize' in cfg.head_name



    def call(self, images):
        base_model_output = self.base_model(images)

        base_model_output_pooled = self.spatial_pooling(base_model_output)
        batch_embedding = self.embedding_head(base_model_output_pooled )
        if self.l2_embedding:
            reutrn_batch_embedding = tf.nn.l2_normalize(batch_embedding, -1)
        else:
            reutrn_batch_embedding = batch_embedding
        return reutrn_batch_embedding

