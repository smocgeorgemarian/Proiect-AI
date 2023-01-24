import threading
from threading import Thread

from algorithm.Algorithm import NO_RUNS
from algorithm.BPAlgorithm import BPAlgorithm
from algorithm.HCAlgorithm import HCAlgorithm
from input.InputLoader import InputLoader
from model.ModelFactory import ModelFactory
from output.PrettyPrinter import PrettyPrinter


def get_experiment_label(algorithm_classes):
    experiment_label = ""
    for alg_cls in algorithm_classes:
        if experiment_label != "":
            experiment_label += "+"
        if alg_cls == BPAlgorithm:
            experiment_label += "BP"
        else:
            experiment_label += "HC"
    return experiment_label


def main(algorithm_classes, lock):
    model = ModelFactory.create_model(is_random=False)
    batch_size = NO_RUNS // len(algorithm_classes)
    experiment_label = get_experiment_label(algorithm_classes)

    for algo_index, algo in enumerate(algorithm_classes):
        start_index = algo_index * batch_size
        algorithm = algo(model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                         final_data=final_data,
                         test_data=test_data, no_iterations=batch_size, experiment_label=experiment_label,
                         start_index=start_index, lock=lock)
        model = algorithm.run()


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = InputLoader.load()
    final_data = []
    test_data = []
    plot_final_data = []
    plot_test_data = []

    n_repetitions = 3
    threads = []
    lock = threading.Lock()
    for algorithm_class in [(BPAlgorithm,), (HCAlgorithm,), (HCAlgorithm, BPAlgorithm), (BPAlgorithm, HCAlgorithm)]:
        for repetition in range(n_repetitions):
            t = Thread(target=main, args=(algorithm_class, lock))
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    PrettyPrinter.print_lineplot(test_data, final_data)
    PrettyPrinter.print_boxplot(test_data, final_data)
