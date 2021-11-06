import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import optimize
import checkNNGradients

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


def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    
    m = X.shape[0]

    Theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1) ], (num_ocultas, (num_entradas + 1)))
    Theta2 = np.reshape(params_rn[ num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

    Delta1 = np.zeros(Theta1.shape)
    Delta2 = np.zeros(Theta2.shape)

    print(Delta1.shape)
    print(Delta2.shape)

    A1, A2, h = propaga_red(X, Theta1, Theta2)

    coste = 0

    for k in range(m):
        a1k = A1[k, :]  # (401,)
        a2k = A2[k, :]  # (26,)
        hk = h[k, :]    # num neuronas ultima capa
        yk = y[k]

        d3k = hk - yk   # error (10,)
        d2k = np.dot(Theta2.T, d3k) * (a2k * (1 - a2k)) # (26,)



def main():
    weights = loadmat('ex4weights.mat')
    Theta1, Theta2 = weights['Theta1'], weights['Theta2']
    X, y = load_data()
    print(coste_red(X, y,Theta1, Theta2))
    print(coste_red_regularizado(X, y, Theta1, Theta2, 1))
    print(Theta1.shape)
    print(Theta2.shape)
    # TODO sacar los tamaños limpios
    backprop(np.concatenate([Theta1.ravel(), Theta2.ravel()]), Theta1.shape[1]-1, Theta2.shape[1]-1, 10, X, y, 1)


main()
