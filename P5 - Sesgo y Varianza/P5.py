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
    data = loadmat('ex5data1.mat')
    X = data['X']
    m = X.shape[0]
    Y = data['y']
    Y = Y[:, 0]
    X_val = data['Xval']
    Y_val = data['yval']
    Y_val = Y_val[:, 0]
    X_test = data['Xtest']
    Y_test = data['ytest']
    Y_test = Y_test[:, 0]
    Theta = np.array([1, 1])
    X = np.hstack([np.ones([X.shape[0], 1]), X])
    X_val = np.hstack([np.ones([X_val.shape[0], 1]), X_val])

    print(coste_regularizado(Theta, X, Y, 1))
    print(gradiente_regularizado(Theta, X, Y, 1))
    res = scipy.optimize.minimize(minimize_this_pls, Theta, args=(X, Y, 0),
                                  jac=True, method='TNC')

    # plot_line(X[:, 1:], Y, res.x)
    Errors = np.empty((m, 2))
    print(Errors.shape)
    sliceSize = np.arange(1, m)
    print(sliceSize)
    for i in sliceSize:
        Theta = np.array([1, 1])
        res = scipy.optimize.minimize(minimize_this_pls, Theta, args=(X[0:i], Y[0:i], 0),
                                      jac=True, method='TNC')
        err = coste_lineal(res.x, X[0:i], Y[0:i])
        errVal = coste_lineal(res.x, X_val, Y_val)
        Errors[i-1] = np.array([err, errVal])


    plot_line(sliceSize, Errors[:-1, 0])
    plot_line(sliceSize, Errors[:-1, 1])
    plt.show()

main()
