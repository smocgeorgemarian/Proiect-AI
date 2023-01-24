import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class PrettyPrinter:
    @staticmethod
    def print_lineplot(test_data, final_data):
        df = pd.DataFrame(final_data)
        ax = sns.lineplot(x='iteration', y='result', hue='experiment', data=df)
        ax.set_title("Train accuracy")
        plt.show()

        df1 = pd.DataFrame(test_data)
        ax1 = sns.lineplot(x='iteration', y='result', hue='experiment', data=df1)
        ax1.set_title("Test accuracy")
        plt.show()

    @staticmethod
    def print_boxplot(test_data, final_data):
        df = pd.DataFrame(final_data)
        ax = sns.boxplot(x='experiment', y='result', data=df[df.iteration == df.iteration.max()])
        ax.set_title("Train accuracy")
        plt.show()

        df1 = pd.DataFrame(test_data)
        ax1 = sns.boxplot(x='experiment', y='result', data=df1[df1.iteration == df1.iteration.max()])
        ax1.set_title("Test accuracy")
        plt.show()
