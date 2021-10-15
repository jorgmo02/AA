from pandas.io.parsers import read_csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def carga_csv(filename):
    valores = read_csv(filename, header=None).to_numpy()
    return valores.astype(float)


def pinta_frontera_recta(X, Y, Theta):
    plt.figure()
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    h = sigmoide(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(Theta))
    h = h.reshape(xx1.shape)

    # el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.savefig("frontera.png")
    plt.close()

def sigmoide(z):
    return (1 / (1 + np.e ** -z))


def coste(Theta, X, Y):
    # X : (m, (n+1))
    # Theta : ((n+1),)
    m = np.shape(X)[0]
    G = sigmoide(np.matmul(X, Theta))
    a = np.transpose(np.log(G)) * Y + np.transpose(np.log(1 - G)) * (1 - Y)
    return np.sum(-a) / m


def gradiente(Theta, X, Y):
    m = np.shape(X)[0]
    G = sigmoide(np.matmul(X, Theta))
    a = np.matmul(np.transpose(X), G - Y)
    return a / m

def pass_or_fail(x):
    return 1 if x >= 0.5 else 0

def porcentaje_aciertos(Theta, X, Y):
    m = np.shape(X)[0]
    values = sigmoide(np.matmul(X, Theta))

    #TODO poner bonito y vectorizar
    for i in range(m):
        if values[i] >= 0.5:
            values[i] = 1
        else:
            values[i] = 0

    success = np.sum(values == Y)

    print("{} aciertos".format(success))
    return success/m*100




def main():
    datos = carga_csv('ex2data1.csv')
    X = datos[:, :-1]  # Todas las columnas excepto la última
    Y = datos[:, -1]  # la ultima columna

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    X = np.hstack([np.ones([m, 1]), X])

    Theta = np.zeros(np.shape(X)[1])

    result = optimize.fmin_tnc(func=coste, x0=Theta, fprime=gradiente, args=(X, Y),messages=0)
    theta_opt = result[0]

    # print('theta optimo {}'.format(theta_opt))
    # print(coste(theta_opt, X, Y))
    # print(gradiente(theta_opt, X, Y))
    # pinta_frontera_recta(X, Y, theta_opt)
    porcentaje =  porcentaje_aciertos(theta_opt, X, Y)
    print(porcentaje)

main()
