import os
import glob
import random
import sklearn
import farmhash
import keras_cv_attention_models

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm.notebook import *
from sklearn.neighbors import *
from sklearn.decomposition import *
from sklearn.model_selection import *

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
    tpu = None
    gpus = tf.config.experimental.list_logical_devices("GPU")
    
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    tf.config.set_soft_device_placement(True)

    print('Running on TPU ', tpu.master())
elif len(gpus) > 0:
    strategy = tf.distribute.MirroredStrategy(gpus)
    print('Running on ', len(gpus), ' GPU(s) ')
else:
    strategy = tf.distribute.get_strategy()
    print('Running on CPU')

print("Number of accelerators: ", strategy.num_replicas_in_sync)

AUTO = tf.data.experimental.AUTOTUNE

config = {
    "seed": 1213,

    "lr": 1e-5,
    "epochs": 20,
    "batch_size": 16 * strategy.num_replicas_in_sync,

    "n_classes": 1000,
    "image_size": [224, 224, 3],
    "hashLength": 1024,

    "data_paths": ['gs://kds-e8b0d5b8e7554df732ca06246dd948f7e744976ca4296eae0d9cac1f', 'gs://kds-8dc2956bd73898b8aadf8ca42ec97eeba443d4d670d1ec141c6399cd', 'gs://kds-1b23b91731c9030c64b4f626012b744adb670233c3a2ae41d83c7f52', 'gs://kds-2460e8e8b49c81a43fa134284b2ec3a71ec165c3e37697807d294ed6', 'gs://kds-842e24bb6b7c321b2ca4cbdbec128929d9b9380eba1f73050c140159', 'gs://kds-dcbf94f355c745f7bb5d51ca5f70d7e3d96d51b004ba73705b2515ed', 'gs://kds-9d88eab50167d0cebc19a36a7247df5c7bb6b6145e2bd4f86176e9e9', 'gs://kds-4de88b574eca43548876201c0a9d9078e08e80cf808e612f9fc3ffc0'],
    "save_path": "./",
    "backbones": ["EfficientNetV2M", "beit.BeitV2BasePatch16", "davit.DaViT_S"]
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(config["seed"])

def intToHash(x):
    return list(map(int, [i for i in bin(x)[3:]]))

def train_transform(image, label):
    image = tf.expand_dims(image, axis=0)
    
    if (tf.random.uniform([1]) < 0.5):
        image = tf.image.random_flip_left_right(image)
    if (tf.random.uniform([1]) < 0.2):
        image = tf.image.random_brightness(image, 0.08)
    # if (tf.random.uniform([1]) < 0.3):
    #     image = tfa.image.rotate(image, 1.0*tf.random.uniform([1]) - 0.5)
    return tf.squeeze(image, axis=0), label

def margin_format(image, label):
    return {'image': image, 'label': label}, label

def onehot(data, label):
    label = tf.one_hot(label, tf.constant(config["n_classes"]), axis =-1, dtype=tf.float32)
    return data, label

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels = 3)
    image = tf.image.resize_with_pad(image, target_width = config["image_size"][0], target_height = config["image_size"][1])
    return image

def read_tfrecord(example):
    TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['label'], dtype = tf.int64)
    return image, label

def load_dataset(filenames, ordered):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False 
    
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls = AUTO) 
    return dataset

def get_train_dataset(filenames):
    dataset = load_dataset(filenames, ordered = False).repeat().shuffle(config["seed"])

    dataset = dataset.map(train_transform, num_parallel_calls = AUTO)
    dataset = dataset.map(onehot, num_parallel_calls = AUTO)
    dataset = dataset.map(margin_format, num_parallel_calls = AUTO)

    dataset = dataset.batch(config["batch_size"])
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_valid_dataset(filenames):
    dataset = load_dataset(filenames, ordered = True)

    dataset = dataset.map(onehot, num_parallel_calls = AUTO)
    dataset = dataset.map(margin_format, num_parallel_calls = AUTO)

    dataset = dataset.batch(config["batch_size"])
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_test_dataset(filenames):
    dataset = load_dataset(filenames, ordered = True)

    dataset = dataset.map(margin_format, num_parallel_calls = AUTO)

    dataset = dataset.batch(config["batch_size"])
    dataset = dataset.prefetch(AUTO) 
    return dataset

class Hash(tf.keras.layers.Layer):
    def __init__(self, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
    def call(self, inputs, training):
        if training:
            return tf.math.sigmoid(self.scale*tf.nn.l2_normalize(inputs, axis = 1))
            # return tf.math.sigmoid(inputs)
        else:
            return tf.math.round(tf.math.sigmoid(self.scale*tf.nn.l2_normalize(inputs, axis = 1)))
            # return tf.math.round(tf.math.sigmoid(inputs))
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'scale': self.scale,
        })
        return config

class Margin(tf.keras.layers.Layer):
    def __init__(self, num_classes, margin = 0.3, scale=64, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.num_classes, input_shape[0][-1]), initializer='glorot_uniform', trainable=True)

    def cosine(self, feature):
        x = tf.nn.l2_normalize(feature, axis=1)
        w = tf.nn.l2_normalize(tf.math.sigmoid(200*tf.nn.l2_normalize(self.W, axis = 1)), axis=1)
        cos = tf.matmul(x, tf.transpose(w))
        return cos

    def logits(self, feature, labels):
        cosine = self.cosine(feature)
        mr = tf.random.normal(shape = tf.shape(cosine), mean = self.margin, stddev = 0.1*self.margin)
        theta = tf.acos(tf.clip_by_value(cosine, -1, 1))
        cosine_add = tf.math.cos(theta + mr)

        mask = tf.cast(labels, dtype=cosine.dtype)
        logits = mask*cosine_add + (1-mask)*cosine
        return tf.nn.l2_normalize(logits, axis = 1)

    def call(self, inputs, training):
        feature, labels = inputs

        if training:
            logits = self.logits(feature, labels)
        else:
            logits = self.cosine(feature)
        return logits*self.scale

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

def model_factory(backbones, n_classes):
    image = tf.keras.layers.Input(shape = (None, None, 3), dtype=tf.uint8, name = 'image')
    label = tf.keras.layers.Input(shape = (), name = 'label', dtype = tf.int64)

    features = [get_backbone(backbone, image) for backbone in backbones]
    headModel = tf.keras.layers.Concatenate(name = "concat")(features)
    headModel = tf.keras.layers.Dense(config["hashLength"], activation = "linear", name = "feature")(headModel)
    headModel = Hash(name = "hash")(headModel)

    margin = Margin(num_classes = n_classes, name = "margin")([headModel, label])
    output = tf.keras.layers.Softmax(dtype=tf.float32, name = "output")(margin)

    model = tf.keras.models.Model(inputs = [image, label], outputs = [output])
    return model

class Evaluation(tf.keras.callbacks.Callback):
    def __init__(self, g_data, q_data):
        self.g_data = g_data
        self.q_data = q_data

    def bitdigest(self, org_digest, digest):
        return [org_digest[i] + "".join(map(lambda v: str(int(v)), x)) for i, x in enumerate(digest.tolist())]

    def to_id(self, ids):
        return [str(x) + ".jpg" for x in ids]

    def process_gallery(self, model, epoch, steps = None):
        feature, g_id = model.predict(self.g_data, verbose = 1, steps = steps)

        df = pd.DataFrame()
        df["image_id"] = self.to_id(g_id)
        df["hashcode"] = self.bitdigest([""]*len(feature), feature)
        df.to_csv(f"{config['save_path']}G_{epoch}.csv", index = False)

    def process_query(self, model, epoch, steps = None):
        feature, q_id = model.predict(self.q_data, verbose = 1, steps = steps)

        df = pd.DataFrame()
        df["image_id"] = self.to_id(q_id)
        df["hashcode"] = self.bitdigest([""]*len(feature), feature)
        df.to_csv(f"{config['save_path']}Q_{epoch}.csv", index = False)

    def gen_sub(self, epoch):
        return os.popen(f"python3 generate_submit_csv.py --gallery {config['save_path']}G_{epoch}.csv --query {config['save_path']}Q_{epoch}.csv --submit {config['save_path']}S_{epoch}.csv").read()

    def on_epoch_end(self, epoch, logs={}):
        model = tf.keras.models.Model(inputs = self.model.inputs,
                                        outputs = [self.model.get_layer('hash').output, self.model.inputs[1]])

        self.process_query(model, epoch)
        self.process_gallery(model, epoch)
        self.gen_sub(epoch)

class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self, path):
        self.path = path

    def on_epoch_end(self, epoch, logs={}):
        self.model.save(self.path + "model.keras")

# DATA_FILENAMES = glob.glob(config["data_path"] + "iBioTrain/iBioTrain/*.tfrec")
# TEST_Q_FILENAMES = glob.glob(config["data_path"] + "FGVC11*Query*.tfrec")
# TEST_G_FILENAMES = glob.glob(config["data_path"] + "FGVC11*Gallery*.tfrec")

# TRAINING_FILENAMES, VALIDATION_FILENAMES = train_test_split(DATA_FILENAMES, test_size=0.02, random_state = config["seed"])

# sample_dataset = get_test_dataset(random.choices(DATA_FILENAMES, k = 1))
# train_dataset = get_train_dataset(TRAINING_FILENAMES)
# valid_dataset = get_valid_dataset(VALIDATION_FILENAMES)
# test_G_dataset = get_test_dataset(sorted(TEST_G_FILENAMES))
# test_Q_dataset = get_test_dataset(sorted(TEST_Q_FILENAMES))

DATA_FILENAMES = []
TEST_G_FILENAMES = []
TEST_Q_FILENAMES = []

for gcs_path in config["data_paths"]:
    DATA_FILENAMES += tf.io.gfile.glob(gcs_path + '/*BIO*.tfrec')
    TEST_G_FILENAMES += tf.io.gfile.glob(gcs_path + '/FGVC10*Test_G*.tfrec')
    TEST_Q_FILENAMES += tf.io.gfile.glob(gcs_path + '/FGVC10*Test_Q*.tfrec')

TRAINING_FILENAMES, VALIDATION_FILENAMES = train_test_split(DATA_FILENAMES, test_size=0.02, random_state = config["seed"])

sample_dataset = get_test_dataset(random.choices(DATA_FILENAMES, k = 1))
train_dataset = get_train_dataset(TRAINING_FILENAMES)
valid_dataset = get_valid_dataset(VALIDATION_FILENAMES)
test_G_dataset = get_test_dataset(sorted(TEST_G_FILENAMES))
test_Q_dataset = get_test_dataset(sorted(TEST_Q_FILENAMES))

with strategy.scope():
    model = model_factory(backbones = config["backbones"],
                          n_classes = config["n_classes"])

    optimizer = tf.keras.optimizers.Adam(learning_rate = config["lr"])
    model.compile(optimizer = optimizer,
                  loss = [tf.keras.losses.CategoricalCrossentropy()],
                  metrics = [tf.keras.metrics.CategoricalAccuracy(name = "ACC@1"),
                             tf.keras.metrics.TopKCategoricalAccuracy(k = 10, name = "ACC@10"),
                             tf.keras.metrics.TopKCategoricalAccuracy(k = 50, name = "ACC@50"),
                             ])
    savemodel = SaveModel(path = config['save_path'])
    evaluation = Evaluation(test_G_dataset, test_Q_dataset)

H = model.fit(train_dataset, verbose = 1,
              validation_data = valid_dataset,
              callbacks = [savemodel, evaluation],
              steps_per_epoch = 2000,
              epochs = config["epochs"])