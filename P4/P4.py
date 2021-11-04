import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import optimize

def load_data(file='ex4data1.mat'):
    data = loadmat(file)
    y = data['y']
    y = y[:, 0]
    X = data['X']
    return X, y


def sigmoide(z):
    return (1 / (1 + np.e ** -z))


def propaga_red(X, Theta1, Theta2):
    m = X.shape[0]

    a1 = np.hstack([np.ones([m  , 1]), X])
    z2 = np.dot(a1, Theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoide(z2)])
    z3 = np.dot(a2, Theta2.T)
    h = sigmoide(z3)

    return a1, a2, h

def safe_log(n):
    return np.log(n + 1e-6)

def one_hot_y(X, y, num_labels=10):
    m = len(y)
    input_size = X.shape[1]
    y = (y - 1)
    y_onehot = np.zeros((m, num_labels))  # 5000 x 10
    for i in range(m):
        y_onehot[i][y[i]] = 1

    return y_onehot

def coste_red(X, Y, Theta1, Theta2):
    m = X.shape[0]
    A1, A2, h = propaga_red(X, Theta1, Theta2)
    hot_y = one_hot_y(X, Y, 10)

    coste = np.sum((-hot_y * safe_log(h)) - ((1-hot_y) * safe_log(1-h)))
    return coste / m

def coste_red_regularizado(X, Y, Theta1, Theta2, reg):
    m = X.shape[0]
    coste_sin_regularizar = coste_red(X, Y, Theta1, Theta2)
    a = np.sum(Theta1[1:]**2)
    b = np.sum(Theta2[1:]**2)

    return coste_sin_regularizar + reg/(2*m) * (a+b)

def main():
    weights = loadmat('ex4weights.mat')
    Theta1, Theta2 = weights['Theta1'], weights['Theta2']
    X, y = load_data()
    print(coste_red(X, y,Theta1, Theta2))
    print(coste_red_regularizado(X, y, Theta1, Theta2, 1))


main()
