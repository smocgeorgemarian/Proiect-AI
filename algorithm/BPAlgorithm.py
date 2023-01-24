import keras

from algorithm.Algorithm import Algorithm
from algorithm.MyCallback import MyCallback


class BPAlgorithm(Algorithm):
    def __init__(self, model, x_train, y_train, x_test, y_test, final_data, test_data, no_iterations, experiment_label,
                 start_index):
        super().__init__(model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                         final_data=final_data, test_data=test_data, no_iterations=no_iterations,
                         experiment_label=experiment_label, start_index=start_index)

    def run(self):
        opt = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        history = self.model.fit(self.x_train, self.y_train, epochs=self.no_iterations, batch_size=10,
                                 callbacks=[MyCallback(model=self.model, x_test=self.x_test, y_test=self.y_test,
                                                       test_data=self.test_data, label=self.experiment_label,
                                                       start_index=self.start_index)])
        for it_index, it_acc in enumerate(history.history['accuracy'], start=0):
            self.final_data.append({
                'experiment': self.experiment_label,
                'iteration': self.start_index + it_index,
                'result': it_acc
            })

        # add results for boxplot
        # if self.plot_final_data:
        #     self.plot_final_data.append({
        #         'experiment': self.experiment_label,
        #         'result': history.history['accuracy'][-1]
        #     })
        # if self.plot_test_data:
        #     self.plot_test_data.append({
        #         'experiment': self.experiment_label,
        #         'result': self.model.evaluate(self.x_test, self.y_test, verbose=0)
        #     })
        return self.model

    def __str__(self):
        return "BP"

    def __repr__(self):
        return "BP"
