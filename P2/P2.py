from pandas.io.parsers import read_csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocatkor


def carga_csv(filename):
    valores = read_csv(filename, header=None).to_numpy()
    return valores.astype(float)


def pinta_frontera_recta(X, Y, Theta):
    return None
    # # TODO vectorizar esto
    # plt.figure()
    # x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    # x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    # xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    # # TODO por que
    # h = sigmoide(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    #                    xx1.ravel(),
    #                    xx2.ravel()].dot(Theta))
    # h = h.reshape(xx1.shape)
    #
    # plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')


def visualizacion(X, Y, Theta=None):
    pos = np.where(Y == 1)
    plt.scatter(X[pos, 1], X[pos, 2], marker='+', c='k', label='Admitted')
    pos = np.where(Y == 0)
    plt.scatter(X[pos, 1], X[pos, 2], marker='o', c='lime', label='Not admitted')
    if (Theta is not None):
        one,other = 1,50
        h1 = np.sum(X[one, :] * Theta)
        h2 = np.sum(X[other, :] * Theta)
        plt.plot([X[one, 1], sigmoide(h1)], [X[other, 1], sigmoide(h2)])

    plt.legend()
    plt.show()


def sigmoide(z):
    return (1 / (1 + np.e ** -z))


def coste(Theta, X, Y):
    m = np.shape(X)[0]
    G = sigmoide(np.matmul(X, Theta))
    a = np.transpose(np.log(G)) * Y + np.transpose(np.log(1 - G)) * (1 - Y)
    return np.sum(-a) / m


def gradiente(Theta, X, Y):
    m = np.shape(X)[0]
    G = sigmoide(np.matmul(X, Theta))
    a = np.matmul(np.transpose(X), G - Y)
    return a / m


def main():
    datos = carga_csv('ex2data1.csv')
    X = datos[:, :-1]  # Todas las columnas excepto la última
    Y = datos[:, -1]  # la ultima columna

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    X = np.hstack([np.ones([m, 1]), X])

    Theta = np.zeros(np.shape(X)[1])
    print(coste(Theta, X, Y))
    print(gradiente(Theta, X, Y))

    result = optimize.fmin_tnc(func=coste, x0=Theta, fprime=gradiente, args=(X, Y))
    theta_opt = result[0]

    print('theta optimo {}'.format(theta_opt))
    visualizacion(X, Y, theta_opt)


main()
