class Algorithm:
    def __init__(self, model, x_train, y_train, x_test, y_test, final_data, test_data, no_iterations, experiment_label,
                 start_index):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.final_data = final_data
        self.test_data = test_data
        self.no_iterations = no_iterations
        self.experiment_label = experiment_label
        self.start_index = start_index

    def run(self):
        pass
