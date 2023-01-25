import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class PrettyPrinter:
    @staticmethod
    def move_into_dir(**kwargs):
        dir_name = "_".join([f"{key}_{value}" for key, value in kwargs.items()])

        expected_path = os.path.join(".", dir_name)
        if not os.path.exists(expected_path):
            os.mkdir(expected_path)
        os.chdir(dir_name)

    @staticmethod
    def print_lineplot(test_data, final_data, **kwargs):
        os.chdir("storage")
        PrettyPrinter.move_into_dir(**kwargs)

        suffix = ".png"
        for title, filename, data in [("Train accuracy", f"line_train{suffix}", final_data),
                                      ("Test accuracy", f"line_test{suffix}", test_data)]:
            df = pd.DataFrame(data)
            ax = sns.lineplot(x='iteration', y='result', hue='experiment', data=df)
            ax.set_title(title)
            plt.savefig(filename)
            plt.show()

    @staticmethod
    def print_boxplot(test_data, final_data):
        suffix = ".png"

        for title, filename, data, must_filter in \
                [("Train accuracy", f"box_train{suffix}", final_data, False),
                 ("Test accuracy", f"box_test{suffix}", test_data, False),
                 ("Train accuracy - No HC Included", f"box_train_no_hc{suffix}", final_data, True),
                 ("Test accuracy - No HC Included", f"box_test_no_hc{suffix}", test_data, True)]:

            df = pd.DataFrame(data)
            if must_filter:
                ax = sns.boxplot(x='experiment', y='result',
                                 data=df[(df.iteration == df.iteration.max()) & (df.experiment != 'HC')])
            else:
                ax = sns.boxplot(x='experiment', y='result', data=df[df.iteration == df.iteration.max()])
            ax.set_title(title)
            plt.savefig(filename)
            plt.show()
