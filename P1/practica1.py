from pandas.io.parsers import read_csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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
    # plt.show()
    plt.savefig("apartado1_line.png")


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
    # plt.show()
    plt.savefig("apartado1_mesh.png")

def show_contour(data):
    #TODO preguntar por logspace
    plt.contour(data[0],data[1],data[2],np.logspace(-2,3,20),colors='blue')
    # plt.scatter(data[0], data[1])
    # plt.contour(data[0],data[1],data[2],colors='blue')
    # plt.show()
    plt.savefig("apartado1_contour.png")

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

def gradiente_it(X, Y, Theta, alpha):
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = np.dot(X, Theta)
    Aux = (H-Y)

    for i in range(n):
        Aux_i = Aux * X[:, i]        
        Theta[i] -= (alpha/m) * Aux_i.sum()     
    return Theta

def gradiente_vec(X, Y, Theta, alpha):
    NuevaTheta = Theta
    m = np.shape(X)[0]
    H = np.dot(X, Theta)
    return Theta - (alpha/m) * np.dot(np.transpose(X), (H-Y))

def descenso_gradiente_multiple(X, Y, alpha=0.01, iteraciones=1500):
    Theta = np.zeros(np.shape(X)[1])
    costes = np.zeros(iteraciones)
    for i in range(iteraciones):
        costes[i] = coste_vec(X, Y, Theta)
        Theta = gradiente_it(X, Y, Theta, alpha)
    
    # Devolveremos todo el proceso para poder comparar distintos
    # Factores de aprendizaje
    return costes, Theta


def ec_normal(X, Y):    
    transX = np.transpose(X)
    XTX = np.dot(transX, X)
    invXT = np.dot(np.linalg.pinv(XTX), transX)
    return np.dot(invXT, Y)


def apartado_2():
    datos = carga_csv('ex1data2.csv')
    mat_norm, mu, sigma = normaliza_matriz(datos)
    X = mat_norm[:, :-1] #Todas las columnas excepto la ultima
    Y = mat_norm[:, -1] #La ultima columna
    m = np.shape(X)[0]
    X = np.hstack([np.ones([m, 1]), X])
    plt.figure()
    
    Alphas = [(0.01,'lime'),(0.1,'blue'),(0.3,'indigo'),(0.03,'teal')]
    for alpha, color in Alphas:
        costes, Theta = descenso_gradiente_multiple(X, Y, alpha,iteraciones=500)
        plt.scatter(np.arange(np.shape(costes)[0]),costes,c=color,label='alpha {}'.format(alpha))
        
    plt.legend()
    plt.savefig("descenso_gradiente.png")

    ejemplo = [1650, 3]
    ejemplo_norm = (ejemplo - mu[:2]) / sigma[:2] #Normalizamos los datos
    ejemplo_norm = np.hstack([[1],ejemplo_norm]) #Añadimos un 1
    prediccion = np.sum(Theta * ejemplo_norm) #Multiplicamos elemento a elemnto
    print(prediccion*sigma[-1] + mu[-1]) #Escalamos el resultado

def apartado_2_2():
    datos = carga_csv('ex1data2.csv')
    ejemplo = [[1, 1650, 3]]
    X = datos[:, :-1] #Todas las columnas excepto la ultima
    Y = datos[:, -1] #La ultima columna
    m = np.shape(X)[0]
    X = np.hstack([np.ones([m, 1]), X])
    Thetas = ec_normal(X, Y)
    print(np.shape(X))
    print(np.shape(ejemplo))
    print(np.shape(Thetas))
    prediccion = np.sum(Thetas * ejemplo)
    print(prediccion)


def main():
    apartado_1()
    apartado_2()
    apartado_2_2()


main()
