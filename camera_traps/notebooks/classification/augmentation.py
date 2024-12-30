import tensorflow as tf
from keras.engine import base_layer


class RandomColorDistortion(base_layer.BaseRandomLayer):
    """
    Custom Keras layer for random color distortion in images.

    This layer applies random brightness and saturation adjustments to input images during training. It also provides
    an option to convert the images to grayscale with a specified probability.
    """

    def __init__(self, brightness_max_delta, saturation_delta, hue_max_delta, contrast_delta, **kwargs):
        super(RandomColorDistortion, self).__init__(**kwargs)
        self.brightness_max_delta = brightness_max_delta
        self.saturation_delta = saturation_delta
        self.hue_max_delta = hue_max_delta
        self.contrast_delta = contrast_delta

    def get_config(self):
        config = {
            "brightness_max_delta": self.brightness_max_delta,
            "saturation_delta": self.saturation_delta,
            "hue_max_delta": self.hue_max_delta,
            "contrast_delta": self.contrast_delta,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def call(self, inputs, training=True):
        if not training:
            return inputs

        inputs = tf.image.random_brightness(inputs, self.brightness_max_delta)
        inputs = tf.image.random_saturation(inputs, self.saturation_delta[0], self.saturation_delta[1])
        inputs = tf.image.random_hue(inputs, self.hue_max_delta)
        inputs = tf.image.random_contrast(inputs, self.contrast_delta[0], self.contrast_delta[1])

        return inputs
