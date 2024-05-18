import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm.notebook import *
from sklearn.neighbors import *
from sklearn.model_selection import *

from data import *
from model import *
from callbacks import *

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
    "batch_size": 8 * strategy.num_replicas_in_sync,

    "n_classes": 1000,
    "image_size": [224, 224, 3],
    "hashLength": 1024,

    "data_paths": ['gs://kds-acbcddd3b90f580dfd4fafd69255ed010bd28a253db452df5915e894', 'gs://kds-06c4c2825b46920e15bf4e8bbd35c1369d01e18665126a38911f9ce4', 'gs://kds-f1f10543e44076f688b6b5591f3d6f8becd7686f65203742aff9fe20', 'gs://kds-61659c280c4a4796313f550e048e5ed00901e04f761d14de93a52684', 'gs://kds-1b5dd9fe16b875f388d2e7a543a140b244b7790277a28da630e83a14', 'gs://kds-db85b5e5bac887fe81d166b7c06550f2ecaa110d95a9b66c47cb4197', 'gs://kds-26b4770c0a699451c074121eae9e28c79153f9fac52069293ecfa4bc', 'gs://kds-ce197112423113b75efe0787828235d71379d11c1a5d0b6f175e0112'],
    "save_path": "./",
    "backbones": ["EfficientNetV2S", "beit.BeitV2BasePatch16"]
}

seed_everything(config["seed"])

DATA_FILENAMES = []
TEST_G_FILENAMES = []
TEST_Q_FILENAMES = []

for gcs_path in config["data_paths"]:
    DATA_FILENAMES += tf.io.gfile.glob(gcs_path + '/*BIO*.tfrec')
    TEST_G_FILENAMES += tf.io.gfile.glob(gcs_path + '/FGVC11*Test_G*.tfrec')
    TEST_Q_FILENAMES += tf.io.gfile.glob(gcs_path + '/FGVC11*Test_Q*.tfrec')

TRAINING_FILENAMES, VALIDATION_FILENAMES = train_test_split(DATA_FILENAMES, test_size=0.02, random_state = config["seed"])

train_dataset = get_train_dataset(TRAINING_FILENAMES, config["batch_size"], config["seed"], AUTO)
valid_dataset = get_valid_dataset(VALIDATION_FILENAMES, config["batch_size"], config["seed"], AUTO)
test_G_dataset = get_test_dataset(sorted(TEST_G_FILENAMES), config["batch_size"], config["seed"], AUTO)
test_Q_dataset = get_test_dataset(sorted(TEST_Q_FILENAMES), config["batch_size"], config["seed"], AUTO)

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
              steps_per_epoch = 50000,
              epochs = config["epochs"])