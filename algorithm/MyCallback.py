import tensorflow


class MyCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, model, x_test, y_test, test_data, label, start_index, lock):
        super().__init__()
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.test_data = test_data
        self.start_index = start_index
        self.label = label
        self.lock = lock

    def on_epoch_end(self, epoch, logs=None):
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        weights = self.model.get_weights()

        biases = weights[1]
        with self.lock:
            self.test_data.append({
                'experiment': self.label,
                'iteration': self.start_index + epoch,
                'result': accuracy
            })
