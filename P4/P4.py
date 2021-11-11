import numpy as np
import scipy.optimize
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import optimize
from checkNNGradients import checkNNGradients
from displayData import displayData

def load_data(file='ex4data1.mat'):
    data = loadmat(file)
    y = data['y']
    y = y[:, 0]
    X = data['X']
    return X, y


def sigmoide(z):
    return 1 / (1 + np.e ** -z)


# devuelve los resultados de la red
def propaga_red(X, Theta1, Theta2):
    m = X.shape[0]

    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, Theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoide(z2)])
    z3 = np.dot(a2, Theta2.T)
    h = sigmoide(z3)

    return a1, a2, h

def safe_log(n):
    #TODO comentar que esto debería ser mas chiquito
    return np.log(n + 1e-7)

def one_hot_y(X, y, num_labels=10):
    m = len(y)
    y = (y - 1)
    y_onehot = np.zeros((m, num_labels))  # 5000 x 10
    for i in range(m):
        y_onehot[i][y[i]] = 1

    return y_onehot

def coste_red(X, Y, Theta1, Theta2):
    m = X.shape[0]
    A1, A2, h = propaga_red(X, Theta1, Theta2)

    coste = np.sum((-Y * safe_log(h)) - ((1-Y) * safe_log(1-h)))
    return coste / m

def coste_red_regularizado(X, Y, Theta1, Theta2, reg):
    m = X.shape[0]
    coste_sin_regularizar = coste_red(X, Y, Theta1, Theta2)
    a = np.sum(Theta1[1:]**2)
    b = np.sum(Theta2[1:]**2)

    return coste_sin_regularizar + reg/(2*m) * (a+b)

def regulariza_gradiente(grad, m , reg, Theta):
    guarda = grad[0]
    grad = grad + (reg/m * Theta)
    grad[0] = guarda
    return grad

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    
    m = X.shape[0]

    Theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1) ], (num_ocultas, (num_entradas + 1)))
    Theta2 = np.reshape(params_rn[ num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

    Delta1 = np.zeros(Theta1.shape)
    Delta2 = np.zeros(Theta2.shape)


    A1, A2, h = propaga_red(X, Theta1, Theta2)

    for k in range(m):
        # Salida de la primera capa ejemplo k
        a1k = A1[k, :]  # (401,)
        # Salida de la segunda capa para el ejemplo k
        a2k = A2[k, :]  # (26,)
        # Salida de la capa final para el ejemplo k
        hk = h[k, :]    # num neuronas ultima capa (10,)
        yk = y[k]       # resultado real del ejemplo k

        # TODO preguntar si el error no esta normalizado
        d3k = hk - yk   # error capa final (10,)
        d2k = np.dot(Theta2.T, d3k) * (a2k * (1 - a2k))  # error capa 2 (26,), se multiplica para ponderar
        # d1k no existe porque no hay un resultado en esa capa aún, son los parámetros

        Delta1 = Delta1 + np.dot(d2k[1:, np.newaxis], a1k[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3k[:, np.newaxis], a2k[np.newaxis, :])


    grad1 = Delta1 / m
    grad2 = Delta2 / m
    grad1 = regulariza_gradiente(grad1, m, reg, Theta1)
    grad2 = regulariza_gradiente(grad2, m, reg, Theta2)

    return coste_red_regularizado(X, y, Theta1, Theta2, 1)#, np.concatenate([np.ravel(grad1), np.ravel(grad2)])




def main():
    weights = loadmat('ex4weights.mat')
    Theta1, Theta2 = weights['Theta1'], weights['Theta2']
    X, y = load_data()
    y = one_hot_y(X, y, 10)

    reg = 1
    params = np.concatenate([Theta1.ravel(), Theta2.ravel()])
    # coste, grad = backprop(params, X.shape[1], Theta1.shape[0], Theta2.shape[0], X, y, reg)
    # checkNNGradients(backprop, reg)

    epsilon = 0.12;
    pesos = np.random.uniform(-epsilon, epsilon, params.shape[0])

    res = optimize.minimize(backprop, x0=pesos,
                            args=(X.shape[1], Theta1.shape[0], Theta2.shape[0], X, y, reg),
                            options={'maxiter': 70})
    print(res.x)

main()
