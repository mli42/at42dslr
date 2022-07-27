#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import utils
import argparse
from typing import List, Tuple

class MyLogRegression():

    def __init__(self, alpha: float, max_iter: int):
        self.alpha = alpha
        self.max_iter = max_iter
        print(f"MyLR: Using {self.alpha = }, and {self.max_iter = }")

    @staticmethod
    def cost_(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-15) -> float:
        """Computes the mean squared error of two non-empty numpy.ndarray.
            The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.ndarray, a vector.
            y_hat: has to be an numpy.ndarray, a vector.
        Returns:
            The cross-entropy loss of the two vectors as a float.
        """
        if y.shape != y_hat.shape:
            return None
        y = y + eps
        m = y.shape[0]
        ones = np.ones(y.shape)

        j_elem = y * np.log(y_hat) + (ones - y) * np.log(ones - y_hat)
        j_elem /= -m
        return np.sum(j_elem)

    def plot_hypo(self, x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, label: str) -> None:
        plt.figure()
        plt.title(f"Data repartition between dataset and prediction model for '{label}'")
        plt.plot(x, y, "o")
        plt.plot(x, y_hat, "o")
        plt.legend([
            *[f"Dataset data for '{string}' feature" for string in utils.FEATURES],
            *[f"Prediction data for '{string}' feature" for string in utils.FEATURES],
        ])

    def simple_plot(self, title: str, plot: List[float], legend: str, axes_labels: Tuple) -> None:
        plt.figure()
        plt.title(title)
        plt.plot(plot)
        plt.legend([legend])
        plt.xlabel(axes_labels[0])
        plt.ylabel(axes_labels[1])

    def minmax(self, x: np.ndarray) -> np.ndarray:
        """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
        Args:
            x: has to be an numpy.ndarray, m * 1.
        Returns:
            x' as a numpy.ndarray, m * 1.
        """
        x = x.copy()
        for i in range(x.shape[1]):
            x[..., i] = utils.minmax(x[..., i], np.min(x[..., i]), np.max(x[..., i]))
        return x

    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes a gradient vector from three non-empty numpy.ndarray
        Args:
            x: has to be a numpy.ndarray, a matrix of dimension m * 1.
            y: has to be a numpy.ndarray, a vector of dimension m * 1.
            theta: has to be a numpy.ndarray, a 2 * 1 vector.
        Returns:
            The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        """
        m = x.shape[0]
        y_hat = utils.predict(x, self.theta)
        x_prime = utils.add_intercept(x)
        nabla_j = x_prime.T.dot(y_hat - y) / m
        return nabla_j

    def fit(self, x: np.ndarray, y: np.ndarray, args, label: str) -> np.ndarray:
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: a vector of dimension m * n
            y: a vector of dimension m * 1
        Returns:
            Trained theta
        """
        print(f"==> Training for label '{label}'...")
        alpha = self.alpha
        self.theta = utils.get_default_theta()
        if x.shape[0] != y.shape[0] or self.theta.shape != (x.shape[1] + 1, 1) or self.max_iter <= 0:
            return None
        m = x.shape[0]
        norm_x = self.minmax(x)
        losses = []
        accuracies = []

        for i in range(self.max_iter):
            gradient = self.gradient(norm_x, y)
            self.theta -= gradient

            y_hat = utils.predict(norm_x, self.theta)
            running_loss = MyLogRegression.cost_(y, y_hat)
            losses.append(running_loss)

            correct = np.sum(y == np.rint(y_hat))
            running_accuracy = 100 * correct / m
            accuracies.append(running_accuracy)

            if args.show and i % 100 == 0:
                self.plot_hypo(x, y, y_hat, label)
                plt.show()

        print(f"Last accuracy: {accuracies[-1]}")
        print(f"Last loss: {losses[-1]}")

        if args.accuracy:
            self.simple_plot(f"Train accuracy for '{label}'", accuracies, 'Accuracy', ('Iteration', 'Accuracy'))
        if args.loss:
            self.simple_plot(f"Train loss for '{label}'", losses, 'Loss', ('Iteration', 'Loss'))
        if args.repartition:
            self.plot_hypo(x, y, y_hat, label)
        if args.accuracy or args.loss or args.repartition:
            plt.show()

        return self.theta

def one_vs_all(args) -> None:
    mylr = MyLogRegression(alpha=args.alpha, max_iter=args.max_iter)
    x, y = utils.get_data("./datasets/train.csv")
    theta = {}

    for label in np.unique(y):
        y_one_vs_all = (y == label).astype(int)
        one_theta = mylr.fit(x, y_one_vs_all, args, label)
        theta[label] = one_theta.flatten().tolist()

    utils.save_theta({ utils.LABEL_FEATURE: theta })


def main():
    DEFAULT_ALPHA = 1e-1
    DEFAULT_ITER = int(7e+3)

    parser = argparse.ArgumentParser(description='Train model with logistic regression')
    parser.add_argument('--alpha', action='store', default=DEFAULT_ALPHA, type=float,
        help=f'define learning rate (default: {DEFAULT_ALPHA})')
    parser.add_argument('--max_iter', action='store', default=DEFAULT_ITER, type=int,
        help=f'define number of iterations (default: {DEFAULT_ITER})')
    parser.add_argument('--show', action='store_true',
        help='display plots during gradient descent')
    parser.add_argument('--accuracy', action='store_true', help='plot the accuracy')
    parser.add_argument('--loss', action='store_true', help='plot the loss')
    parser.add_argument('-r', dest='repartition', action='store_true', help='plot the data repartition')
    args = parser.parse_args()

    one_vs_all(args)


if __name__ == "__main__":
    main()
