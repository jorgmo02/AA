from scipy.io import loadmat
import numpy as np

def coste_regularizado(Theta, X, Y, lamb):
    m = X.shape[0]
    reg = lamb/(2*m) * np.sum(Theta[1:]**2)
    return coste_lineal(Theta, X, Y) + reg


def coste_lineal(Theta, X, Y):
    m = X.shape[0]
    H = np.dot(X, Theta.T)
    sigma = np.sum((H - Y) ** 2)
    return sigma / (2 * m)


def main():
    data = loadmat('ex5data1.mat')
    X = data['X']
    Y = data['y']
    X_val = data['Xval']
    Y_val = data['yval']
    X_test = data['Xtest']
    Y_test = data['ytest']
    Theta = {1, 1}
    print(coste_regularizado(Theta, X, Y, 1))

    print('a')

main()