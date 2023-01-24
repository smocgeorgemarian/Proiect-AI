import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class PrettyPrinter:
    @staticmethod
    def print(test_data, final_data):
        df = pd.DataFrame(final_data)
        _ = sns.lineplot(x='iteration', y='result', hue='experiment', data=df)
        plt.show()

        df1 = pd.DataFrame(test_data)
        ax = sns.lineplot(x='iteration', y='result', hue='experiment', data=df1)
        ax.set_title("Test accuracy")
        plt.show()
