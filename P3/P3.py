import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy import optimize


def visualizar_random(X):
    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1, 20).T)
    plt.axis('off')
    plt.show()


def sigmoide(z):
    return (1 / (1 + np.e ** -z))


def coste(Theta, X, Y):
    # X : (m, (n+1))
    # Theta : ((n+1),)
    m = np.shape(X)[0]
    G = sigmoide(np.matmul(X, Theta))
    a = np.transpose(np.log(G)) * Y + np.transpose(np.log(1 - G)) * (1 - Y)
    return np.sum(-a) / m


def coste_regularizado(Theta, X, Y, lamb):
    m = np.shape(X)[0]
    sin_regularizar = coste(Theta, X, Y)
    # TODO preguntar por el termino de regularizacion en el coste == preguntar por si aquí es Theta[1:] o Theta a secas
    regularizacion = np.sum(Theta[1:] ** 2)
    r = (lamb * regularizacion) / (2 * m)
    return sin_regularizar + r


def gradiente(Theta, X, Y):
    m = np.shape(X)[0]
    G = sigmoide(np.matmul(X, Theta))
    a = np.matmul(np.transpose(X), G - Y)
    return a / m


def lambda_for_gradient(x, m, lamb):
    return (lamb / m) * x


def gradiente_regularizado(Theta, X, Y, lamb):
    grad = gradiente(Theta, X, Y)
    g_0 = grad[0]
    regularizador = (lamb / np.shape(X)[0]) * Theta
    grad = grad + regularizador
    grad[0] = g_0

    return grad


def oneVsAll(X, y, num_etiquetas, reg):
    m = X.shape[0]
    n = X.shape[1]
    Theta = np.zeros((num_etiquetas, n))
    m = X.shape[0]
    # poly = PolynomialFeatures(6)
    # poly_X = poly.fit_transform(X)

    for etiqueta in range(num_etiquetas):
        busca = 10 if (etiqueta == 0) else etiqueta
        Theta_k = np.zeros(n)
        coste_r = coste_regularizado(Theta_k, X, y, reg)
        gradiente_r = gradiente_regularizado(Theta_k, X, y, reg)

        result = optimize.fmin_tnc(func=coste_regularizado, x0=Theta_k, fprime=gradiente_regularizado, args=(X, (y == busca)*1, reg),messages=0)
        Theta[etiqueta] = result[0]

    return Theta

def ten_at_zero(x):
    return 10 if x == 0 else x

def comprueba_aciertos(X, y, Theta):
    k = Theta.shape[0]
    acc = 0
    m = X.shape[0]
    # Tantas filas como casos y tantas columnas como etiquetas
    results = np.zeros((X.shape[0], Theta.shape[0]))

    print("Theta shape{}".format(Theta.shape))
    print("X shape{}".format(X.shape))
    print("res shape{}".format(results.shape))

    for etiqueta in range(k):
        results[:, etiqueta] = sigmoide(np.matmul(X, Theta[etiqueta]))
        # num = etiqueta if (etiqueta != 0) else 10

    print(results[1,:])
    predicciones = np.argmax(results, axis=1)
    print(predicciones[:100])

    ten_at_zero_v = np.vectorize(ten_at_zero)
    predicciones = ten_at_zero_v(predicciones)
    acc = np.sum(predicciones == y)

    return (acc/m)*100


def main():
    data = loadmat('ex3data1.mat')
    y = data['y']
    y = y[:, 0]
    X = data['X']
    X = np.hstack([np.ones([X.shape[0], 1]), X])

    reg = 0.1
    Theta = oneVsAll(X, y, 10, reg)
    print(Theta.shape)
    print(comprueba_aciertos(X,y, Theta))


main()
