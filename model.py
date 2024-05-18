import tensorflow as tf
import keras_cv_attention_models

class Margin(tf.keras.layers.Layer):
    def __init__(self, num_classes, margin = 0.1, scale=32, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.num_classes, input_shape[0][-1]), initializer='glorot_uniform', trainable=True)

    # def build_hash(self, x, scale = 10.0):
    #     return tf.nn.sigmoid(scale*tf.nn.l2_normalize(x, axis = 1))

    # def hamming_distance(self, feature):
    #     x = self.build_hash(feature, self.scale)
    #     w = self.build_hash(self.W, self.scale)

    #     x = tf.tile(tf.expand_dims(feature, 2), [1, 1, self.W.shape[0]])
    #     w = tf.transpose(self.W)
    #     return tf.clip_by_value(tf.reduce_sum(tf.math.abs(x - w), axis = 1), 1e-4, 48)

    # def logits_hamming(self, feature, labels):
    #     distance = self.hamming_distance(feature)
    #     mr = tf.random.normal(shape = tf.shape(distance), mean = self.margin, stddev = 0.1*self.margin)
    #     distance_add = distance + mr
    #     mask = tf.cast(labels, dtype=distance.dtype)
    #     logits = self.scale/(mask*distance_add + (1-mask)*distance)
    #     return logits

    def distance(self, feature):
        x = tf.nn.l2_normalize(feature, axis=1)
        w = tf.nn.l2_normalize(self.W, axis=1)

        x = tf.tile(tf.expand_dims(x, 2), [1, 1, self.W.shape[0]])
        w = tf.transpose(w)
        return tf.reduce_sum( tf.math.pow( x - w, 2 ), axis=1)

    def logits_distance(self, feature, labels):
        distance = self.distance(feature)
        mr = tf.random.normal(shape = tf.shape(distance), mean = self.margin, stddev = 0.1*self.margin)
        distance_add = distance + mr
        mask = tf.cast(labels, dtype=distance.dtype)
        logits = self.scale/(mask*distance_add + (1-mask)*distance)
        return logits

    def cosine(self, feature):
        x = tf.nn.l2_normalize(feature, axis=1)
        w = tf.nn.l2_normalize(self.W, axis=1)
        cos = tf.matmul(x, tf.transpose(w))
        return cos

    def logits_cosine(self, feature, labels):
        cosine = self.cosine(feature)
        mr = tf.random.normal(shape = tf.shape(cosine), mean = self.margin, stddev = 0.1*self.margin)
        theta = tf.acos(tf.clip_by_value(cosine, -1, 1))
        cosine_add = tf.math.cos(theta + mr)

        mask = tf.cast(labels, dtype=cosine.dtype)
        logits = mask*cosine_add + (1-mask)*cosine
        return logits*self.scale

    def call(self, inputs, training):
        feature, labels = inputs

        if training:
            logits = self.logits_distance(feature, labels)
        else:
            logits = self.cosine(feature)
        return logits

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'scale': self.scale,
            'margin': self.margin,
            'num_classes': self.num_classes,
        })
        return config
    
def get_backbone(backbone_name, x):
    if hasattr(tf.keras.applications, backbone_name):
        headModel = tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="tf"))(x)
        return tf.keras.layers.GlobalAveragePooling2D()(getattr(tf.keras.applications, backbone_name)(weights = "imagenet", include_top = False)(headModel))
    else:
        backbone = getattr(getattr(keras_cv_attention_models, backbone_name.split(".")[0]), backbone_name.split(".")[1])(num_classes=0)
        headModel = tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"))(x)
        backbone.trainable = True
        if "beit" in backbone_name:
            return backbone(headModel)
        return tf.keras.layers.GlobalAveragePooling2D()(backbone(headModel))

def model_factory(backbones, embDim, n_classes):
    image = tf.keras.layers.Input(shape = (None, None, 3), dtype=tf.uint8, name = 'image')
    label = tf.keras.layers.Input(shape = (), name = 'label', dtype = tf.int64)

    features = [get_backbone(backbone, image) for backbone in backbones]
    headModel = tf.keras.layers.Concatenate(name = "concat")(features)
    headModel = tf.keras.layers.Dense(embDim, activation = "linear", name = "feature")(headModel)

    margin = Margin(num_classes = n_classes, name = "margin")([headModel, label])
    output = tf.keras.layers.Softmax(dtype=tf.float32, name = "output")(margin)

    model = tf.keras.models.Model(inputs = [image, label], outputs = [output])
    return model