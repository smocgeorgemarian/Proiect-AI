import keras

from algorithm.MyCallback import MyCallback


class BPAlgorithm:
    def __init__(self, model, x_train, y_train, x_test, y_test, final_data, test_data):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.final_data = final_data
        self.test_data = test_data

    def run(self):
        opt = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        history = self.model.fit(self.x_train, self.y_train, epochs=150, batch_size=10,
                                 callbacks=[MyCallback(self.model, self.x_test, self.y_test, self.test_data)])
        for it_index, it_acc in enumerate(history.history['accuracy'], start=1):
            self.final_data.append({
                'experiment': 'BP',
                'iteration': it_index,
                'result': it_acc
            })

        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
