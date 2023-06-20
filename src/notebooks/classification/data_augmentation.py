import numpy as np
import tensorflow as tf
from keras.engine import base_layer
from keras.layers.preprocessing import preprocessing_utils as utils


def convert_inputs(inputs, dtype=None):
    if isinstance(inputs, dict):
        raise ValueError(
            "This layer can only process a tensor representing an image or "
            f"a batch of images. Received: type(inputs)={type(inputs)}."
            "If you need to pass a dict containing "
            "images, labels, and bounding boxes, you should "
            "instead use the preprocessing and augmentation layers "
            "from `keras_cv.layers`. See docs at "
            "https://keras.io/api/keras_cv/layers/"
        )
    inputs = utils.ensure_tensor(inputs, dtype=dtype)
    return inputs


class RandomColorDistortion(base_layer.BaseRandomLayer):
    def __init__(self, brightness_delta, saturation_delta, **kwargs):
        super(RandomColorDistortion, self).__init__(**kwargs)
        self.brightness_delta = brightness_delta
        self.saturation_delta = saturation_delta

    def call(self, inputs, training=True):
        inputs = convert_inputs(inputs, self.compute_dtype)

        if not training:
            return inputs

        brightness = np.random.uniform(
            self.brightness_delta[0], self.brightness_delta[1])
        saturation = np.random.uniform(
            self.saturation_delta[0], self.saturation_delta[1])

        inputs = tf.image.adjust_brightness(inputs, brightness)
        inputs = tf.image.adjust_saturation(inputs, saturation)

        if tf.random.uniform([]) < 0.5:
            inputs = tf.image.rgb_to_grayscale(inputs)
            inputs = tf.tile(inputs, [1, 1, 1, 3])

        return inputs
