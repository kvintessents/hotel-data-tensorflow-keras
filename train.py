import tensorflow as tf
import numpy as np

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os
from getHotelData import getHotelData
from getInputModel import getInputModel, plotModel
from getModel import getModel

checkpoint_path = "weights/weights.ckpt"

# Data
(train_data, train_labels), (test_data, test_labels) = getHotelData()
(processing_model, input_symbolics) = getInputModel(train_data)

hotel_model = getModel(processing_model, input_symbolics)

plotModel(hotel_model)

train_features_dict = { name: np.array(value) for name, value in train_data.items() }
test_features_dict = { name: np.array(value) for name, value in test_data.items() }

# Train model
history = hotel_model.fit(
    x = train_features_dict,
    y = train_labels,
    epochs = 10,
    validation_data = (
        test_features_dict,
        test_labels
    )
)

hotel_model.save('hotel_model')

# Evaluate
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
