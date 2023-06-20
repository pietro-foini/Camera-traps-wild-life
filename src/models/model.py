import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB5


def efficientnet_b0(num_classes, input_shape: tuple = (224, 224, 3)):
    """
    Build convolutional neural network model structure for image classification.

    :param num_classes: the number of classes the model should be able to classify
    :param input_shape: the input shape allowed by the model
    :return: the keras model structure
    """
    inputs = layers.Input(shape=input_shape)

    # Pretrained model.
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights.
    for layer in model.layers:
        layer.trainable = False

    # Rebuild top.
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    # Classification layer.
    outputs = layers.Dense(num_classes, activation="softmax", name="prediction")(x)

    # Model structure.
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")

    return model


def efficientnet_b5(num_classes, input_shape: tuple = (224, 224, 3)):
    """
    Build convolutional neural network model structure for image classification.

    :param num_classes: the number of classes the model should be able to classify
    :param input_shape: the input shape allowed by the model
    :return: the keras model structure
    """
    inputs = layers.Input(shape=input_shape)

    # Pretrained model.
    model = EfficientNetB5(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights.
    for layer in model.layers:
        layer.trainable = False

    # Rebuild top.
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    # Classification layer.
    outputs = layers.Dense(num_classes, activation="softmax", name="prediction")(x)

    # Model structure.
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")

    return model
