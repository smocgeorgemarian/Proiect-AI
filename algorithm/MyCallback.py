import tensorflow


class MyCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, model, x_test, y_test, test_data):
        super().__init__()
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        self.test_data.append({
            'experiment': 'BP',
            'iteration': epoch,
            'result': accuracy
        })
