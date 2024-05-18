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
    "embDim": 1024,

    "data_paths": ['gs://kds-3f5ce2e56559d17c577c6d9f0c1acd2179da3bc0c1f87ae24d66f478', 'gs://kds-6d0cfb649153417b33f05a9913c59a358a3c03706a8fd6c5f523b8fc', 'gs://kds-0a7e8b42af254bd7711b5670f5aeb0eeda8d169841722ea59680d3d4', 'gs://kds-d27f1d2f32cff13b0ce97a6b912eb44a68e8ee3128504dcb52e23a3e', 'gs://kds-7a36360336e83a60d74c0196158fed9d6fa70cd342dc96383b7f6e2a', 'gs://kds-f99762a40174159b2d5b1caedff9d3e42758db9134c67405c68ce52d', 'gs://kds-af8ef10c715e13cb364f2655d5188aedfbb874b71e250d2773dade89', 'gs://kds-b0b80433ab5beabf70aca68659672322018f637039ae27fbd9ee9bf1'],
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
    model = model_factory(embDim = config["embDim"],
                          backbones = config["backbones"],
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