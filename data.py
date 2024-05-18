import random
import numpy as np
import tensorflow as tf

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

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
    label = tf.one_hot(label, tf.constant(1000), axis =-1, dtype=tf.float32)
    return data, label

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels = 3)
    image = tf.image.resize_with_pad(image, target_width = 224, target_height = 224)
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

def load_dataset(filenames, ordered, AUTO):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False 
    
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls = AUTO) 
    return dataset

def get_train_dataset(filenames, batch, seed, AUTO):
    dataset = load_dataset(filenames, ordered = False).repeat().shuffle(seed)

    dataset = dataset.map(train_transform, num_parallel_calls = AUTO)
    dataset = dataset.map(onehot, num_parallel_calls = AUTO)
    dataset = dataset.map(margin_format, num_parallel_calls = AUTO)

    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_valid_dataset(filenames, batch, seed, AUTO):
    dataset = load_dataset(filenames, ordered = True)

    dataset = dataset.map(onehot, num_parallel_calls = AUTO)
    dataset = dataset.map(margin_format, num_parallel_calls = AUTO)

    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_test_dataset(filenames, batch, seed, AUTO):
    dataset = load_dataset(filenames, ordered = True)

    dataset = dataset.map(margin_format, num_parallel_calls = AUTO)

    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(AUTO) 
    return dataset