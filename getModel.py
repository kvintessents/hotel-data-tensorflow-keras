import tensorflow as tf
from tensorflow.keras import layers

def getModel(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dropout(0.2),
        layers.Dense(64),
        layers.Dense(128),
        layers.Dense(1)
    ])

    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(
        loss = tf.losses.BinaryCrossentropy(from_logits = True),
        optimizer = tf.optimizers.Adam(),
        metrics = ['accuracy']
    )

    return model