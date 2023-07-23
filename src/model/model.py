import tensorflow as tf


def efficientnet_b0(num_classes: int, input_shape: tuple):
    """
    Build convolutional neural network model structure for image classification.

    :param num_classes: the number of classes the model should be able to classify
    :param input_shape: the input shape allowed by the model
    :return: the keras model structure
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Pretrained model.
    model = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Rebuild top.
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = tf.keras.layers.BatchNormalization()(x)

    # Classification layer.
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="prediction")(x)

    # Model structure.
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")

    return model
