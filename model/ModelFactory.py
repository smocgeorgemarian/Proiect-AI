from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

RELU = 'sigmoid'


class ModelFactory:
    @staticmethod
    def create_model(is_random=False):
        model = None
        if not is_random:
            model = Sequential()
            model.add(Dense(16, input_shape=(4,), activation=RELU))
            model.add(Dense(8, activation=RELU))
            model.add(Dense(3, activation='softmax'))
        return model
