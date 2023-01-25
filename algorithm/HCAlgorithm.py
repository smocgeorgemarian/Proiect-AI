import keras
import numpy as np

from algorithm.Algorithm import Algorithm


class HCAlgorithm(Algorithm):
    def __init__(self, model, x_train, y_train, x_test, y_test, final_data, test_data, no_iterations, experiment_label,
                 start_index, lock):
        super().__init__(model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                         final_data=final_data, test_data=test_data, no_iterations=no_iterations,
                         experiment_label=experiment_label, start_index=start_index, lock=lock)

    def run(self):
        if self.start_index == 0:
            opt = keras.optimizers.Adam(learning_rate=0.01)
            self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        old_loss, old_acc = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        for it_index in range(self.no_iterations):
            delta_weights = [np.random.uniform(-1, 1, w.shape) for w in self.model.get_weights()]
            old_weights = list(self.model.get_weights())
            result_weights = [np.zeros(shape=w.shape) for w in self.model.get_weights()]
            for w_index, weight in enumerate(old_weights):
                result_weights[w_index] = (delta_weights[w_index] + weight) % 1

            self.model.set_weights(delta_weights)
            new_loss, new_acc = self.model.evaluate(self.x_train, self.y_train, verbose=0)
            if new_loss <= old_loss:
                old_loss = new_loss
                old_acc = new_acc
            else:
                self.model.set_weights(old_weights)

            self.tmp_final_data.append({
                'experiment': self.experiment_label,
                'iteration': self.start_index + it_index,
                'result': old_acc
            })

            _, test_acc = self.model.evaluate(self.x_test, self.y_test, verbose=1)
            self.tmp_test_data.append({
                'experiment': self.experiment_label,
                'iteration': self.start_index + it_index,
                'result': test_acc
            })

        with self.lock:
            self.final_data.extend(self.tmp_final_data)
            self.test_data.extend(self.tmp_test_data)
        self.save_model_if_needed()
        return self.model

    def __str__(self):
        return "HC"

    def __repr__(self):
        return "HC"
