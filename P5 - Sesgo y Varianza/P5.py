import scipy.optimize
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


def plot_decisionboundary(X, Y, Theta, poly):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
                           np.linspace(x2_min, x2_max))

    h = coste_lineal(poly.fit_transform(np.c_[xx1.ravel(),
                                              xx2.ravel()]).dot(Theta))
    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    plt.show()


def h(x, theta):
    return theta[0] + theta[1] * x


def plot_line(X, Y):
    plt.plot(X, Y)


def plot_regression(X, Y, theta):
    min_x = np.min(X)
    max_x = np.max(X)
    min_y = h(min_x, theta)
    max_y = h(max_x, theta)
    plt.plot(X, Y, "x")
    plt.plot([min_x, max_x], [min_y, max_y])
    # plt.savefig("apartado1_line.png")

def plot_polynomial_regression(X, Y, Theta, p):
    Pol_X = polinomiza_atributos(X, p)
    Pol_X = np.hstack([np.ones([Pol_X.shape[0], 1]), Pol_X])
    Norm_Poly_X, media, varianza = normaliza_matriz(Pol_X)
    h = Norm_Poly_X.dot(Theta.T)
    print(Theta)
    print(Norm_Poly_X)
    print(varianza)
    print(media)

    print(h)
    h = h * varianza[1] + media[1]
    print(h)
    plt.scatter(X, Y, marker="x")
    # plt.plot para dibujar puntos conectados, scatter para puntos suelts
    # plot_line(X[:, 1], h)


def coste_regularizado(Theta, X, Y, lamb):
    m = X.shape[0]
    reg = (lamb / (2 * m)) * np.sum(Theta[1:] ** 2)
    return coste_lineal(Theta, X, Y) + reg


def coste_lineal(Theta, X, Y):
    m = X.shape[0]
    H = np.dot(X, np.transpose(Theta))
    sigma = np.sum((H - Y) ** 2)
    return sigma / (2 * m)


def gradiente(Theta, X, Y):
    m = np.shape(X)[0]
    H = np.dot(X, Theta)
    a = np.matmul(np.transpose(X), H - Y)
    return a / m


def gradiente_regularizado(Theta, X, Y, lamb):
    grad = gradiente(Theta, X, Y)
    g_0 = grad[0]
    regularizador = (lamb / np.shape(X)[0]) * Theta
    grad = grad + regularizador
    grad[0] = g_0
    return grad


def minimize_this_pls(Theta, X, Y, lamb):
    return coste_regularizado(Theta, X, Y, lamb), gradiente_regularizado(Theta, X, Y, lamb)


def main():
    # plot_line(X[:, 1:], Y, res.x)
    # Errors = learning_curves(X, Y, X_val, Y_val)
    # learning_curves()
    polynomial_regression()


main()
