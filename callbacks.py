import pickle
import numpy as np
import tensorflow as tf

class Evaluation(tf.keras.callbacks.Callback):
    def __init__(self, g_data, q_data):
        self.g_data = g_data
        self.q_data = q_data

    def process_gallery(self, model, epoch, steps = None):
        feature, g_id = model.predict(self.g_data, verbose = 1, steps = steps)
        feature = np.array(tf.nn.l2_normalize(feature, axis=1)).astype(np.float16)
        return feature, g_id

    def process_query(self, model, epoch, steps = None):
        feature, q_id = model.predict(self.q_data, verbose = 1, steps = steps)
        feature = np.array(tf.nn.l2_normalize(feature, axis=1)).astype(np.float16)
        return feature, q_id

    def on_epoch_end(self, epoch, logs={}):
        model = tf.keras.models.Model(inputs = self.model.inputs,
                                        outputs = [self.model.get_layer('feature').output, self.model.inputs[1]])

        with open(f"data_{epoch}", 'wb') as handle:
            pickle.dump([self.process_query(model, epoch), self.process_gallery(model, epoch), self.process_vgallery(model, epoch)], handle, protocol=pickle.HIGHEST_PROTOCOL)

class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self, path):
        self.path = path

    def on_epoch_end(self, epoch, logs={}):
        self.model.save(self.path + "model.keras")