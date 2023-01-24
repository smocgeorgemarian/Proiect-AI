import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class InputLoader:
    @staticmethod
    def load():
        dataset = sns.load_dataset("iris")
        x = dataset.iloc[:, 0:4].values
        y = dataset.iloc[:, 4].values
        encoder = LabelEncoder()
        y1 = encoder.fit_transform(y)
        y = pd.get_dummies(y1).values
        # x_train, x_test, y_train, y_test
        return train_test_split(x, y, test_size=0.2, random_state=0)
