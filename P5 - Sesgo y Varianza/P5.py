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


def normaliza_matriz(x):
    mu = np.mean(x, axis=0)  # Media de cada columna
    sigma = np.std(x, axis=0)  # Desviacion estandar por columnas, no confundir con la querida std de c++

    return (x - mu) / sigma, mu, sigma


def polinomiza_atributos(X, p):
    Pol = X
    for i in range(2, p + 1):
        Pol = np.hstack([Pol, X ** i])

    return Pol


def load_data(filename='ex5data1.mat'):
    data = loadmat(filename)
    X = data['X']
    Y = data['y']
    Y = Y[:, -1]
    X_val = data['Xval']
    Y_val = data['yval']
    Y_val = Y_val[:, -1]
    X_test = data['Xtest']
    Y_test = data['ytest']
    Y_test = Y_test[:, -1]
    return X, Y, X_val, Y_val, X_test, Y_test


def learning_curves():
    X, Y, X_val, Y_val, X_test, Y_test = load_data()

    X = np.hstack([np.ones([X.shape[0], 1]), X])
    X_val = np.hstack([np.ones([X_val.shape[0], 1]), X_val])
    m = X.shape[0]

    Errors = np.empty((m, 2))
    sliceSize = np.arange(1, m)

    for i in sliceSize:
        Theta = np.array([1, 1])
        res = scipy.optimize.minimize(minimize_this_pls, Theta, args=(X[0:i], Y[0:i], 0),
                                      jac=True, method='TNC')
        err = coste_lineal(res.x, X[0:i], Y[0:i])
        errVal = coste_lineal(res.x, X_val, Y_val)
        Errors[i - 1] = np.array([err, errVal])

    plt.plot(sliceSize, Errors[:-1, 0])
    plt.plot(sliceSize, Errors[:-1, 1])
    plt.show()


def try_funcs():
    X, Y, X_val, Y_val, X_test, Y_test = load_data()
    m = X.shape[0]
    Theta = np.array([1, 1])
    X = np.hstack([np.ones([X.shape[0], 1]), X])

    print(coste_regularizado(Theta, X, Y, 1))
    print(gradiente_regularizado(Theta, X, Y, 1))


def polinomize_and_normalize(M, p):
    Pol_M = polinomiza_atributos(M, p)
    Pol_M, media, varianza = normaliza_matriz(Pol_M)

    Pol_M = np.hstack([np.ones([Pol_M.shape[0], 1]), Pol_M])

    # TODO puede que haga falta hacer hstack de 1 a la media y 0 a la varianza
    # media = np.hstack([np.ones(1), media])
    # varianza = np.hstack([np.zeros(1), varianza])

    return Pol_M, media, varianza


def plot_polynomial_regression(X, Y, Theta, p):
    Norm_Pol_X, mu, sigma = polinomize_and_normalize(X, p)
    h = Norm_Pol_X.dot(Theta.T)

    plotSpace = np.linspace(min(X), max(X), 1700)
    plotSpace_np = polinomiza_atributos(plotSpace, p)
    plotSpace_np = (plotSpace_np - mu) / sigma
    plotSpace_np = np.hstack([np.ones([plotSpace_np.shape[0], 1]), plotSpace_np])

    h_plot = plotSpace_np.dot(Theta.T)

    plt.scatter(X, Y, marker="x", c='red')
    plt.plot(plotSpace, h_plot)  # para dibujar puntos conectados, scatter para puntos suelts


def learning_curves_polynomial():
    X, Y, X_val, Y_val, X_test, Y_test = load_data()
    m = X.shape[0]

    p = 8
    X_np, media, varianza = polinomize_and_normalize(X, p)
    X_val_np, media, varianza = polinomize_and_normalize(X_val, p)

    lamb = 0
    Errors = np.empty((m, 2))
    sliceSize = np.arange(1, m)

    for i in sliceSize:
        Theta = np.ones(X_np.shape[1])
        res = scipy.optimize.minimize(minimize_this_pls, Theta, args=(X_np[0:i], Y[0:i], lamb),
                                      jac=True, method='TNC')
        err = coste_lineal(res.x, X_np[0:i], Y[0:i])
        errVal = coste_lineal(res.x, X_val_np, Y_val)
        Errors[i - 1] = np.array([err, errVal])

    plot_line(sliceSize, Errors[:-1, 0])
    plot_line(sliceSize, Errors[:-1, 1])
    plt.show()


def polynomial_regression():
    X, Y, X_val, Y_val, X_test, Y_test = load_data()
    m = X.shape[0]

    p = 8
    Pol_X = polinomiza_atributos(X, p)
    Norm_Pol_X, media, varianza = normaliza_matriz(Pol_X)

    Norm_Pol_X = np.hstack([np.ones([Norm_Pol_X.shape[0], 1]), Norm_Pol_X])
    Theta = np.ones(Norm_Pol_X.shape[1])

    lamb = 0
    res = scipy.optimize.minimize(minimize_this_pls, Theta, args=(Norm_Pol_X, Y, lamb),
                                  jac=True, method='TNC')

    plot_polynomial_regression(X, Y, res.x, p)
    plt.show()


def choose_lambda():
    X, Y, X_val, Y_val, X_test, Y_test = load_data()
    m = X.shape[0]

    p = 8
    X_np, mu, sigma = polinomize_and_normalize(X, p)

    X_val_np = polinomiza_atributos(X_val, p)
    X_val_np = (X_val_np-mu)/sigma
    X_val_np = np.hstack([np.ones([X_val_np.shape[0], 1]), X_val_np])

    X_test_np = polinomiza_atributos(X_test, p)
    X_test_np = (X_test_np-mu)/sigma
    X_test_np = np.hstack([np.ones([X_test_np.shape[0], 1]), X_test_np])

    lambdas = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    Errors = np.empty((lambdas.shape[0], 2))
    i = 0
    Theta = np.ones(X_np.shape[1])
    for lamb in lambdas:
        Theta = np.ones(X_np.shape[1])
        res = scipy.optimize.minimize(minimize_this_pls, Theta, args=(X_np, Y, lamb),
                                      jac=True, method='TNC')
        err = coste_lineal(res.x, X_np, Y)
        errVal = coste_lineal(res.x, X_val_np, Y_val)
        Errors[i] = np.array([err, errVal])
        i = i + 1

    plt.plot(lambdas, Errors[:, 0], label='Train')
    plt.plot(lambdas, Errors[:, 1], label='Cross Validation')
    plt.legend()
    plt.show()

    res = scipy.optimize.minimize(minimize_this_pls, Theta, args=(X_np, Y, 3),
                                  jac=True, method='TNC')
    print(coste_lineal(res.x, X_test_np, Y_test))


def main():
    # plot_line(X[:, 1:], Y, res.x)
    # Errors = learning_curves(X, Y, X_val, Y_val)
    # learning_curves()
    # polynomial_regression()
    choose_lambda()


main()
