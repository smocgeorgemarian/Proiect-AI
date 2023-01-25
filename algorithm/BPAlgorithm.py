import keras

from algorithm.Algorithm import Algorithm
from algorithm.MyCallback import MyCallback


class BPAlgorithm(Algorithm):
    def __init__(self, model, x_train, y_train, x_test, y_test, final_data, test_data, no_iterations, experiment_label,
                 start_index, lock):
        super().__init__(model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                         final_data=final_data, test_data=test_data, no_iterations=no_iterations,
                         experiment_label=experiment_label, start_index=start_index, lock=lock)

    def run(self):
        if self.start_index == 0:
            opt = keras.optimizers.Adam(learning_rate=0.01)
            self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        history = self.model.fit(self.x_train, self.y_train, epochs=self.no_iterations, batch_size=10, verbose=0,
                                 callbacks=[MyCallback(model=self.model, x_test=self.x_test, y_test=self.y_test,
                                                       test_data=self.tmp_test_data, label=self.experiment_label,
                                                       start_index=self.start_index, lock=self.lock)])
        for it_index, it_acc in enumerate(history.history['accuracy']):
            self.tmp_final_data.append({
                'experiment': self.experiment_label,
                'iteration': self.start_index + it_index,
                'result': it_acc
            })
        with self.lock:
            self.final_data.extend(self.tmp_final_data)
            self.test_data.extend(self.tmp_test_data)

        self.save_model_if_needed()
        return self.model

    def __str__(self):
        return "BP"

    def __repr__(self):
        return "BP"
