from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras import initializers
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed, Flatten
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM


def cnn_lstm(input_shape,  drop_CNN=0, drop_dense=0.2, kernel_size=(3, 3)):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(128, kernel_size=kernel_size,
              padding="same", activation="relu"), input_shape=input_shape))
    model.add(Dropout(drop_CNN))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same')))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(128, kernel_size=kernel_size,
              padding="same", activation="relu")))
    model.add(Dropout(drop_CNN))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same')))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(LSTM(units=128, return_sequences=False))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(drop_dense))
    # model.add(Dense(1, activation='linear'))
    return model
