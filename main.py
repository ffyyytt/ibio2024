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

    "lr": 1e-5,
    "epochs": 10,
    "batch_size": 16 * strategy.num_replicas_in_sync,

    "n_classes": 1000,
    "image_size": [224, 224, 3],

    "data_paths": ['gs://kds-01281822938aaf62ec2b3b62244c0a2be4da014e34fb50259ea6c0a5', 'gs://kds-f5e857da02f49a79e947b7eb0e85ffd3ee7622b75bae7b309f907df0', 'gs://kds-34a27c33c6dd72f07d7c61929e3ba7a641bc8aeb77cac18364fc5098', 'gs://kds-e3f80cdf7e780a3dbe79ea338358e620bc65c5a01c36bb5a0811acf5', 'gs://kds-f89ea4d15874e276588a9a144d9b743c46c99f0f898f2d410661f3f7', 'gs://kds-0eabfdaac2e8422c7cdaeda976ff631bbea9b91f245faa4915216c2c'],
    "save_path": "./",
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

def get_test_dataset(filenames):
    dataset = load_dataset(filenames, ordered = True)

    dataset = dataset.map(margin_format, num_parallel_calls = AUTO)

    dataset = dataset.batch(config["batch_size"])
    dataset = dataset.prefetch(AUTO) 
    return dataset

class Hash(tf.keras.layers.Layer):
    def __init__(self, scale=50, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
    def call(self, inputs, training):
        if training:
            return tf.math.sigmoid(self.scale*tf.nn.l2_normalize(inputs, axis = 1))
        else:
            return tf.math.round(tf.math.sigmoid(self.scale*tf.nn.l2_normalize(inputs, axis = 1)))
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'scale': self.scale,
        })
        return config

class Margin(tf.keras.layers.Layer):
    def __init__(self, num_classes, margin = 2.0, scale=32, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.num_classes, input_shape[0][-1]), initializer='zeros', trainable=False)

    def hamming(self, feature):
        # x = tf.clip_by_value(feature, 0, 1)
        # w = tf.clip_by_value(self.W, 0, 1)

        x = tf.tile(tf.expand_dims(feature, 2), [1, 1, self.W.shape[0]])
        w = tf.transpose(self.W)

        # tf.print(tf.nn.softmax(tf.nn.l2_normalize(48-tf.reduce_sum(tf.math.abs(x - w), axis = 1), axis = 1)*self.scale, axis = 1))
        return 48-tf.reduce_sum(tf.math.abs(x - w), axis = 1)

    def logits(self, feature, labels):
        distance = self.hamming(feature)
        mr = tf.random.normal(shape = tf.shape(distance), mean = self.margin, stddev = 0.1*self.margin)
        distance_add = distance + mr

        mask = tf.cast(labels, dtype=distance.dtype)
        logits = mask*distance + (1-mask)*distance_add
        return tf.nn.l2_normalize(logits, axis = 1)

    def call(self, inputs, training):
        feature, labels = inputs

        if training:
            logits = self.logits(feature, labels)
        else:
            logits = tf.nn.l2_normalize(self.hamming(feature), axis = 1)
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
    headModel = tf.keras.layers.Dense(48, activation = "linear", name = "feature")(headModel)
    headModel = Hash(name = "hash")(headModel)

    margin = Margin(num_classes = n_classes, name = "margin")([headModel, label])
    output = tf.keras.layers.Softmax(dtype=tf.float32, name = "output")(margin)

    model = tf.keras.models.Model(inputs = [image, label], outputs = [output])
    return model

class Evaluation(tf.keras.callbacks.Callback):
    def __init__(self, g_data, q_data):
        self.g_data = g_data
        self.q_data = q_data

    def bitdigest(self, digest):
        return ["".join(map(lambda v: str(int(v)), x)) for x in digest.tolist()]

    def to_id(self, ids):
        return [str(x) + ".jpg" for x in ids]

    def process_gallery(self, model, epoch, steps = None):
        feature, g_id = model.predict(self.g_data, verbose = 1, steps = steps)

        df = pd.DataFrame()
        df["image_id"] = self.to_id(g_id)
        df["hashcode"] = self.bitdigest(feature)
        df.to_csv(f"{config['save_path']}G_{epoch}.csv", index = False)

    def process_query(self, model, epoch, steps = None):
        feature, q_id = model.predict(self.q_data, verbose = 1, steps = steps) 

        df = pd.DataFrame()
        df["image_id"] = self.to_id(q_id)
        df["hashcode"] = self.bitdigest(feature)
        df.to_csv(f"{config['save_path']}Q_{epoch}.csv", index = False)

    def gen_sub(self, epoch):
        return os.Popen(f"python3 generate_submit_csv.py --gallery {config['save_path']}G_{epoch}.csv --query {config['save_path']}Q_{epoch}.csv --submit {config['save_path']}S_{epoch}.csv").read()

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

DATA_FILENAMES = []
TEST_G_FILENAMES = []
TEST_Q_FILENAMES = []

for gcs_path in config["data_paths"]:
    DATA_FILENAMES += tf.io.gfile.glob(gcs_path + '/*BIO*.tfrec')
    TEST_G_FILENAMES += tf.io.gfile.glob(gcs_path + '/*Test_G*.tfrec')
    TEST_Q_FILENAMES += tf.io.gfile.glob(gcs_path + '/*Test_Q*.tfrec')

TRAINING_FILENAMES, VALIDATION_FILENAMES = train_test_split(DATA_FILENAMES, test_size=0.02, random_state = config["seed"])

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
    model.layers[-2].set_weights([features])
    savemodel = SaveModel(path = config['save_path'])
    evaluation = Evaluation(test_G_dataset, test_Q_dataset)

H = model.fit(train_dataset, verbose = 1,
              validation_data = valid_dataset,
              callbacks = [savemodel, evaluation],
              steps_per_epoch = 1000,
              epochs = config["epochs"])