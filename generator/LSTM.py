import keras as ks
import numpy as np
import pandas as pd

def createLSTM(x, y, maxlen, vocabular_size):
    print(">>Creating LSTM model<<")
    model = ks.Sequential()
    model.add(ks.layers.LSTM(128, input_shape=(maxlen, vocabular_size)))
    model.add(ks.layers.Dense(vocabular_size))
    model.add(ks.layers.Activation('softmax'))
    model.summary()
    model.compile(optimizer=ks.optimizers.Adam(0.01), loss='categorical_crossentropy')
    model.fit(x, y, epochs= 5, batch_size= 128)
    print(">>Model created<<")
    return model