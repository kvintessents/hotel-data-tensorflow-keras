import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import datetime
import numpy as np
from getHotelData import getHotelData

def getSymbolicTensor(df):
    inputs = {}

    for name, column in df.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32
        
        inputs[name] = tf.keras.Input(shape = (1,), name = name, dtype = dtype)
    
    return inputs

def concatNumericInputs(inputs, data):
    numeric_inputs = {name: input for name, input in inputs.items() if input.dtype == tf.float32}

    x = layers.Concatenate()(list(numeric_inputs.values()))
    norm = preprocessing.Normalization()
    norm.adapt(np.array(data[numeric_inputs.keys()]))
    all_numeric_inputs = norm(x)

    return [all_numeric_inputs]

def concatCategoricalInputs(inputs, data):
    categorical_inputs = []

    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue
            
        print(name)

        lookup = preprocessing.StringLookup(vocabulary = np.unique(data[name]))
        one_hot = preprocessing.CategoryEncoding(max_tokens = lookup.vocab_size())

        x = lookup(input)
        x = one_hot(x)

        categorical_inputs.append(x)
    
    return categorical_inputs

def plotModel(inputModel, to_file = 'latest_model.png'):
    print('Plotting latest model...')
    tf.keras.utils.plot_model(model = inputModel, rankdir="LR", dpi=72, show_shapes=True, to_file = to_file)

def getInputModel(data):
    inputs = getSymbolicTensor(data)

    numerics = concatNumericInputs(inputs, data)
    categorics = concatCategoricalInputs(inputs, data)

    preprocessed_inputs_cat = layers.Concatenate()(numerics + categorics)

    hotel_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

    return (hotel_preprocessing, inputs) 

if __name__ == '__main__':
    ((train_data, train_labels), (test_data, test_labels)) = getHotelData()
    data = train_data

    model = getInputModel(data)

    data_dict = { name: np.array(value) for name, value in data.items() }
