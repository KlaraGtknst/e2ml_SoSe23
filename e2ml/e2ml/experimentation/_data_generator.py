"""
The following functions serve as blackbox data generator functions.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NOISE = 0.05
THETA = np.random.RandomState(0).randint(10, 100, 16)


def black_box_data_generation(X):
    learning_rate_list = np.linspace(0.05, 1, 20)[X[:, 0]]
    batch_size_list = np.array([16, 32, 64, 128])[X[:, 1]]
    momentum_list = np.linspace(0.05, 1, 20)[X[:, 2]]
    hidden_layer_size_list = np.array([10, 25, 50, 75, 10])[X[:, 3]]
    X, y = make_classification(n_samples=500, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    scores = []
    for learning_rate, batch_size, momentum, hidden_layer_size in zip(
        learning_rate_list, batch_size_list, momentum_list, hidden_layer_size_list
    ):
        mlp = MLPClassifier(
            learning_rate_init=learning_rate,
            batch_size=batch_size,
            hidden_layer_sizes=hidden_layer_size,
            max_iter=1000,
            random_state=0,
        )
        scores.append(mlp.fit(X_train, y_train).score(X_test, y_test))
    scores = np.array(scores)
    return scores


def get_hotellings_experiment_measurements(X):
    y = X @ THETA
    return y + np.random.randn(len(y))


def get_hotellings_experiment_errors(theta_hat):
    return ((theta_hat - THETA) ** 2).sum()


def class_gen(x):
    mean = np.mean(x)
    x_bool = x >= mean
    x_class = [1 if x_bool[i] == True else 0 for i in range(len(x_bool))]
    return x_class


def foo1(x1, x2, x3, x4, x5, x6):
    foo1_ = class_gen(
        x1 + 2 * (x2 + NOISE * np.random.random()) - x3 ** 2 - x4 + 4 * x5 - 2 * x6 ** 2 + NOISE * np.random.random()
    )
    return foo1_


def foo2(x1, x2, x3, x4):
    foo2_ = class_gen(
        2 * x1 ** 4
        + np.abs(x2)
        + NOISE * np.random.random()
        - x3 * (x4 + NOISE * np.random.random())
        + NOISE * np.random.random()
    )
    return foo2_


def foo3(x1, x2):
    foo3_ = class_gen(1 - 4 ** np.abs(x1) + NOISE * np.random.random() + 2 * (x2 + NOISE * np.random.random()) ** 2)
    return foo3_
