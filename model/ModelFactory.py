import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


RELU = 'sigmoid'


class ModelFactory:
    @staticmethod
    def create_model(is_random=False):
        model = Sequential()
        # model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        # model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        # model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

        model.add(Dense(16, input_shape=(4,), activation=RELU))
        model.add(Dense(8, activation=RELU))
        model.add(Dense(3, activation='softmax'))
        return model
