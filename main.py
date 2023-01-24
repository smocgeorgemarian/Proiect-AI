from algorithm.BPAlgorithm import BPAlgorithm
from algorithm.HCAlgorithm import HCAlgorithm
from input.InputLoader import InputLoader
from model.ModelFactory import ModelFactory
from output.PrettyPrinter import PrettyPrinter


def main(algorith_class):
    algorithm = algorith_class(model, x_train, y_train, x_test, y_test, final_data=final_data, test_data=test_data)
    algorithm.run()


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = InputLoader.load()

    for algorithm_class in [HCAlgorithm, BPAlgorithm]:
        final_data = []
        test_data = []
        n_repetitions = 3

        for repetition in range(n_repetitions):
            model = ModelFactory.create_model(is_random=False)
            main(algorithm_class)

        PrettyPrinter.print(test_data, final_data)

