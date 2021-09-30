from pandas.io.parsers import read_csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def carga_csv(filename):
    valores = read_csv(filename, header=None).to_numpy()
    return valores.astype(float)


def h(x, theta):
    return theta[0] + theta[1] * x


def func_coste(X, Y, theta):
    acc = 0
    m = len(X)
    acc = np.sum((h(X, theta) - Y) ** 2)
    return acc / (2 * m)


def plot_line(X, Y, theta):
    min_x = min(X)
    max_x = max(X)
    min_y = h(min_x, theta)
    max_y = h(max_x, theta)
    plt.plot(X, Y, "x")
    plt.plot([min_x, max_x], [min_y, max_y])
    plt.show()


def descenso_gradiente_simple(X, Y, alpha=0.01, iteraciones=1500):
    theta_0 = theta_1 = 0
    m = len(X)
    for _ in range(iteraciones):
        acc_0 = np.sum(h(X, [theta_0, theta_1]) - Y)
        acc_1 = np.sum((h(X, [theta_0, theta_1]) - Y) * X)
        theta_0 = theta_0 - (alpha / m) * acc_0
        theta_1 = theta_1 - (alpha / m) * acc_1
    return [theta_0, theta_1]


def make_grid(t0_range, t1_range, X, Y, step=0.1):
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)

    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)

    Coste = np.empty_like(Theta0)
    #TODO comprobar si se puede limpiar este bucle
    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix, iy] = func_coste(X, Y, [Theta0[ix, iy], Theta1[ix, iy]])

    return [Theta0, Theta1, Coste]


def show_mesh(data):
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(data[0], data[1], data[2], cmap=cm.jet, linewidth=0, antialiased=False)
    plt.show()

def show_contour(data):
    #TODO preguntar por logspace
    plt.contour(data[0],data[1],data[2],np.logspace(-2,3,20),colors='blue')
    # plt.contour(data[0],data[1],data[2],colors='blue')
    plt.show()

def apartado_1():
    datos = carga_csv('ex1data1.csv')
    X = datos[:, 0]
    Y = datos[:, 1]
    theta = descenso_gradiente_simple(X, Y)
    # plot_line(X, Y, theta)
    grid_data = make_grid([-10, 10], [-1, 4], X, Y)
    # show_mesh(grid_data)
    show_contour(grid_data)

def normaliza_matriz(x):
    mu = np.mean(x, axis=0)  # Media de cada columna
    sigma = np.std(x, axis=0)  # Desviacion estandar por columnas, no confundir con la querida std de c++
    x_norm = (x-mu)/sigma

    return x_norm, mu, sigma

def coste_vec(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = (H-Y) ** 2
    return Aux.sum() / (2*len(X))

def descenso_gradiente_multiple(X, Y, alpha=0.01, iteraciones=1500):
    Theta = np.zeros(np.shape(X)[1])
    m = np.shape(X)[0]
    print(m)
    # print(np.shape(Theta))
    for _ in range(iteraciones):
        cost = coste_vec(X, Y, Theta)
        Theta = Theta - (alpha/m) * cost



def apartado_2():
    datos = carga_csv('ex1data2.csv')
    X = datos[:, :-1] #Todas las columnas excepto la ultima
    # print(np.shape(X))
    Y = datos [:, -1] #La ultima columna
    # print(np.shape(Y))
    x_norm, mu, sigma = normaliza_matriz(X)
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    X = np.hstack([np.ones([m, 1]), X])
    alpha = 0.01
    descenso_gradiente_multiple(X, Y)


def main():
    apartado_2()


main()
