import os
import random
import sklearn
import farmhash
import keras_cv_attention_models

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm.notebook import *
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

    "lr": 2e-4,
    "epochs": 5,
    "batch_size": 16 * strategy.num_replicas_in_sync,

    "n_classes": 1000,
    "image_size": [224, 224, 3],

    "data_paths": ['gs://kds-e3f80cdf7e780a3dbe79ea338358e620bc65c5a01c36bb5a0811acf5', 'gs://kds-f5e857da02f49a79e947b7eb0e85ffd3ee7622b75bae7b309f907df0', 'gs://kds-f89ea4d15874e276588a9a144d9b743c46c99f0f898f2d410661f3f7', 'gs://kds-34a27c33c6dd72f07d7c61929e3ba7a641bc8aeb77cac18364fc5098'],
    "save_path": "iBIO2/",
    "backbones": ["EfficientNetV2M", "beit.BeitV2BasePatch16"]
}

def seed_everything(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(config["seed"])

def intToHash(x):
    return list(map(int, [i for i in bin(x)[3:]]))

def genHash():
    s = set()
    features = np.zeros([1000, 48])
    for i in range(1000):
        features[i] = intToHash(farmhash.FarmHash64(str(i) + "seed" + str(config["seed"])))[:48]
        if str(features[i]) in s:
            print("BUG")
        s.add(str(features[i]))
    return features

features = genHash()

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

class Margin(tf.keras.layers.Layer):   
    def __init__(self, num_classes, margin = 0.5, scale=32, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.num_classes, input_shape[0][-1]), initializer='zeros', trainable=False)

    def build_hash(self, x, scale = 10.0):
        x = tf.nn.l2_normalize(x, axis = 1)
        return tf.nn.relu(scale*x)

    def hamming(self, feature):
        x = tf.clip_by_value(self.build_hash(feature, 50), 0, 1)
        w = tf.clip_by_value(self.build_hash(self.W, 50), 0, 1)

        x = tf.tile(tf.expand_dims(x, 2), [1, 1, w.shape[0]])
        w = tf.transpose(w)

        return 48-tf.reduce_sum(tf.math.abs(x - w), axis = 1)
    
    def logits(self, feature, labels):
        distance = self.hamming(feature)
        mr = tf.random.normal(shape = tf.shape(distance), mean = self.margin, stddev = 0.1*self.margin)
        distance_add = distance + mr

        mask = tf.cast(labels, dtype=distance.dtype)
        logits = mask*distance + (1-mask)*distance_add
        return logits

    def call(self, inputs, training):
        feature, labels = inputs

        if training:
            logits = self.logits(feature, labels)
        else:
            logits = self.hamming(feature)
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
    headModel = tf.keras.layers.Concatenate()(features)
    headModel = tf.keras.layers.Dense(48, activation = "linear")(headModel)
    
    margin = Margin(num_classes = n_classes)([headModel, label])
    output = tf.keras.layers.Softmax(dtype=tf.float32)(margin)
    
    model = tf.keras.models.Model(inputs = [image, label], outputs = [output])
    return model

DATA_FILENAMES = []

for gcs_path in config["data_paths"]:
    DATA_FILENAMES += tf.io.gfile.glob(gcs_path + '/*BIO*.tfrec')

TRAINING_FILENAMES, VALIDATION_FILENAMES = train_test_split(DATA_FILENAMES, test_size=0.05, random_state = config["seed"])
print(TRAINING_FILENAMES)
print(VALIDATION_FILENAMES)

train_dataset = get_train_dataset(TRAINING_FILENAMES)
valid_dataset = get_valid_dataset(VALIDATION_FILENAMES)

class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self, path):
        self.path = path

    def on_epoch_end(self, epoch, logs={}):
        self.model.save(self.path + "model.h5")

with strategy.scope():
    model = model_factory(backbones = config["backbones"],
                          n_classes = config["n_classes"])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = config["lr"])
    model.compile(optimizer = optimizer,
                  loss = [tf.keras.losses.CategoricalCrossentropy()],
                  metrics = [tf.keras.metrics.CategoricalAccuracy(name = "ACC@1"), 
                             tf.keras.metrics.TopKCategoricalAccuracy(k = 5, name = "ACC@5"),
                             tf.keras.metrics.TopKCategoricalAccuracy(k = 20, name = "ACC@20"),
                             ])
    model.layers[-2].set_weights([features])
    savemodel = SaveModel(path = config['save_path'])

H = model.fit(train_dataset, verbose = 1,
              validation_data = valid_dataset,
              callbacks = [savemodel],
              steps_per_epoch = 1000,
              epochs = config["epochs"])