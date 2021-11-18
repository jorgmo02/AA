from pandas.io.parsers import read_csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.preprocessing import PolynomialFeatures 

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

def coste_regularizado(Theta, X, Y, lamb):
    m = np.shape(X)[0]
    sin_regularizar = coste(Theta, X, Y)
    #TODO preguntar por el termino de regularizacion en el coste == preguntar por si aquí es Theta[1:] o Theta a secas
    regularizacion = np.sum(Theta[1:] ** 2)
    r = (lamb * regularizacion) / (2 * m)
    return sin_regularizar + r

def gradiente(Theta, X, Y):
    m = np.shape(X)[0]
    G = sigmoide(np.matmul(X, Theta))
    a = np.matmul(np.transpose(X), G - Y)
    return a / m

def lambda_for_gradient(x,m,lamb):
    return (lamb/m)*x

def gradiente_regularizado(Theta, X, Y, lamb):
    grad = gradiente(Theta, X, Y)
    g_0 = grad[0]
    regularizador = (lamb/np.shape(X)[0])*Theta
    grad = grad + regularizador
    grad[0] = g_0
    
    # suma_reg = np.vectorize(lambda_for_gradient)
    # grad = suma_reg(grad[1:],np.shape(X)[0],lamb)
    return grad
    

def pass_or_fail(x):
    return 1 if x >= 0.5 else 0

def porcentaje_aciertos(Theta, X, Y):
    m = np.shape(X)[0]
    values = sigmoide(np.matmul(X, Theta))

    pass_or_fail_v = np.vectorize(pass_or_fail)
    values = pass_or_fail_v(values)

    success = np.sum(values == Y)

    print("{} aciertos".format(success))
    return success/m*100

def regresion_logistica():
    datos = carga_csv('ex2data1.csv')
    X = datos[:, :-1]  # Todas las columnas excepto la última
    Y = datos[:, -1]  # la ultima columna
    m = np.shape(X)[0]
    X = np.hstack([np.ones([m, 1]), X])
    n = np.shape(X)[1]

    Theta = np.zeros(n)

    result = optimize.fmin_tnc(func=coste, x0=Theta, fprime=gradiente, args=(X, Y),messages=0)
    theta_opt = result[0]

    # print('theta optimo {}'.format(theta_opt))
    # print(coste(theta_opt, X, Y))
    # print(gradiente(theta_opt, X, Y))
    # pinta_frontera_recta(X, Y, theta_opt)
    porcentaje =  porcentaje_aciertos(theta_opt, X, Y)
    print(porcentaje)

def plot_decisionboundary(X, Y, Theta, poly):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
                           np.linspace(x2_min, x2_max))
 
    h = sigmoide(poly.fit_transform(np.c_[xx1.ravel(),
                                         xx2.ravel()]).dot(Theta))
    h = h.reshape(xx1.shape)
 
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    plt.show()
    #plt.savefig("boundary.png")


def visualiza_casos(X, Y):
    pos = np.where(Y == 1)
    plt.scatter(X[pos,0],X[pos, 1], marker='+', c='black')
    pos = np.where(Y == 0)
    plt.scatter(X[pos,0],X[pos, 1], marker='o', c='orange')

def regresion_regularizada():
    datos = carga_csv('ex2data2.csv')
    X = datos[:, :-1]  # Todas las columnas excepto la última
    Y = datos[:, -1]  # la ultima columna

    # Mapeo de atributos
    poly = PolynomialFeatures(6)
    poly_X = poly.fit_transform(X)
    m = np.shape(poly_X)[0]
    n = np.shape(poly_X)[1]
    # print(np.shape(poly_X))

    Theta = np.zeros(n)

    Lambdas = np.linspace(0, 5, num=10)
    Lambdas = [1]

    for lamb in Lambdas:
        coste_r = coste_regularizado(Theta,poly_X, Y, lamb)
        gradiente_r = gradiente_regularizado(Theta, poly_X,Y, lamb)
        if lamb == 1:
            print(coste_r)
            print(gradiente_r)

        result = optimize.fmin_tnc(func=coste_regularizado,x0=Theta, fprime=gradiente_regularizado, args=(poly_X,Y, lamb), messages=0)
        theta_opt = result[0]

        plt.figure()
        visualiza_casos(X, Y)
        plot_decisionboundary(X, Y, theta_opt, poly)
        plt.savefig("Samples/Comprobacion_lambda{}.png".format(lamb))
        plt.close()

def main():
    # regresion_logistica()
    regresion_regularizada()
    

main()