import keras
import numpy as np

from algorithm.Algorithm import Algorithm


class HCAlgorithm(Algorithm):
    def __init__(self, model, x_train, y_train, x_test, y_test, final_data, test_data):
        super().__init__(model, x_train, y_train, x_test, y_test, final_data, test_data)

    def run(self):
        opt = keras.optimizers.Adam(learning_rate=0.0001)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        n_iterations = 150
        old_loss, old_acc = self.model.evaluate(self.x_train, self.y_train, verbose=0)

        for it_index in range(n_iterations):
            delta_weights = [np.random.uniform(-1, 1, w.shape) - 0.5 for w in self.model.get_weights()]
            old_weights = list(self.model.get_weights())
            result_weights = [np.zeros(shape=w.shape) for w in self.model.get_weights()]
            for w_index, weight in enumerate(old_weights):
                result_weights[w_index] = (delta_weights[w_index] + weight) % 1

            self.model.set_weights(delta_weights)
            new_loss, new_acc = self.model.evaluate(self.x_train, self.y_train, verbose=1)
            if new_loss <= old_loss:
                old_loss = new_loss
                old_acc = new_acc
            else:
                self.model.set_weights(old_weights)

            self.final_data.append({
                'experiment': 'HC',
                'iteration': it_index,
                'result': old_acc
            })

            _, test_acc = self.model.evaluate(self.x_test, self.y_test, verbose=1)
            self.test_data.append({
                'experiment': 'HC',
                'iteration': it_index,
                'result': test_acc
            })
