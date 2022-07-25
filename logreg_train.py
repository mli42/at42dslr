#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import utils
import argparse
from typing import List

class MyLogRegression():

    def __init__(self, alpha: float, max_iter: int):
        self.alpha = alpha
        self.max_iter = max_iter
        print(f"MyLR: Using {self.alpha = }, and {self.max_iter = }")

    @staticmethod
    def cost_(y: np.ndarray, y_hat: np.ndarray) -> float:
        """Computes the mean squared error of two non-empty numpy.ndarray.
            The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.ndarray, a vector.
            y_hat: has to be an numpy.ndarray, a vector.
        Returns:
            The mean squared error (MSE) of the two vectors as a float.
        """
        if y.shape != y_hat.shape:
            return None
        j_elem = (y_hat - y) ** 2 / y.shape[0]
        return np.sum(j_elem)

    def plot_hypo(self, x: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> None:
        plt.figure()
        # Data repartition
        plt.plot(x, y, "o")
        # Prediction model
        plt.plot(x, y_hat)
        plt.title('Data repartition and predition model')
        plt.legend(['Dataset','Hypothesis'])
        plt.xlabel("Mileage (km)")
        plt.ylabel("Price of car")

    def plot(self, x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, losses: List[float]) -> None:
        self.plot_hypo(x, y, y_hat)

        plt.figure()
        plt.title('Train loss through epochs')
        plt.plot(losses)
        plt.legend(['Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.show()

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

    def fit(self, x: np.ndarray, y: np.ndarray, show_gradient: bool) -> np.ndarray:
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: a vector of dimension m * n
            y: a vector of dimension m * 1
        Returns:
            Trained theta
        """
        alpha = self.alpha
        self.theta = utils.get_default_theta()
        if x.shape[0] != y.shape[0] or self.theta.shape != (x.shape[1] + 1, 1) or self.max_iter <= 0:
            return None
        # norm_x = self.minmax(x)
        norm_x = x
        losses = []
        for i in range(self.max_iter):
            gradient = self.gradient(norm_x, y)
            self.theta -= gradient

            y_hat = utils.predict(norm_x, self.theta)
            running_loss = MyLogRegression.cost_(y, y_hat)
            losses.append(running_loss)

            if show_gradient and i % 100 == 0:
                self.plot_hypo(x, y, y_hat)
                plt.show()

        # self.plot(x, y, y_hat, losses)
        return self.theta

def one_vs_all(args) -> None:
    mylr = MyLogRegression(alpha=args.alpha, max_iter=args.max_iter)
    x, y = utils.get_data("./datasets/train.csv")
    theta = {}

    for feat_value in np.unique(y):
        y_one_vs_all = (y == feat_value).astype(int)
        one_theta = mylr.fit(x, y_one_vs_all, args.show)
        theta[feat_value] = one_theta.flatten().tolist()

    utils.save_theta({ utils.LABEL_FEATURE: theta })


def main():
    DEFAULT_ALPHA = 1e-2
    DEFAULT_ITER = 1000

    parser = argparse.ArgumentParser(description='Train model with logistic regression')
    parser.add_argument('--alpha', action='store', default=DEFAULT_ALPHA, type=float,
        help=f'define learning rate (default: {DEFAULT_ALPHA})')
    parser.add_argument('--max_iter', action='store', default=DEFAULT_ITER, type=int,
        help=f'define number of iterations (default: {DEFAULT_ITER})')
    parser.add_argument('--show', action='store_true',
        help='display plots during gradient descent')
    args = parser.parse_args()

    one_vs_all(args)


if __name__ == "__main__":
    main()
