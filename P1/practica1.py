from pandas.io.parsers import read_csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def carga_csv(filename):
    valores = read_csv(filename,header=None).to_numpy()
    return valores.astype(float)

def punto_en_linea(x,th_0,th_1):
    return th_0 + (x * th_1)

def func_coste(X,Y,theta_0,theta_1):
    m = len(X)
    sum_vec = np.sum((punto_en_linea(X,theta_0,theta_1)-Y)**2)
    return sum_vec/(2*m)
    
def coste(X,Y,Theta):
    H = np.dot(X,Theta)
    Aux = (H-Y) ** 2
    return Aux.sum() / (2*len(X))

def descenso_it(datos,alpha=0.01,iteraciones=1500):
    theta_0 = theta_1 = 0
    X = datos[:,0]
    Y = datos[:,1]
    m = len(datos)
    for _ in range(iteraciones):
        sum_0 = sum_1 = 0
        for i in range(m):
            sum_0 += punto_en_linea(X[i],theta_0,theta_1)-Y[i]
            sum_1 += (punto_en_linea(X[i],theta_0,theta_1)-Y[i]) * X[i]
        theta_0 = theta_0 - (alpha/m) * sum_0
        theta_1 = theta_1 - (alpha/m) * sum_1
        print("th0 {} \t\tth1 {} \tsquared error {}".format(theta_0,theta_1,func_coste(X,Y,theta_0,theta_1)))

    plt.plot(X,Y,"x")
    min_x = min(X)
    max_x = max(X)
    min_y = theta_0 + theta_1 * min_x
    max_y = theta_0 + theta_1 * max_x
    plt.plot([min_x,max_x],[min_y,max_y])
    plt.savefig('resultado.pdf')

def descenso_gradiente(datos,alpha=0.01,iteraciones=1500):
    theta_0 = theta_1 = 0
    X = datos[:,0]
    Y = datos[:,1]
    m = len(datos)
    # TODO mirar si podemos cargarnos este bucle
    for _ in range(iteraciones):
        aux_0 = theta_0 - (alpha/m) * np.sum(punto_en_linea(X,theta_0,theta_1) - Y)
        aux_1 = theta_1 - (alpha/m) * np.sum((punto_en_linea(X,theta_0,theta_1) - Y)*X)
        theta_0 = aux_0
        theta_1 = aux_1
        print("th0 {} th1 {} squared error {}".format(theta_0,theta_1,func_coste(X,Y,theta_0,theta_1)))

    plt.plot(X,Y,"x")
    min_x = min(X)
    max_x = max(X)
    min_y = theta_0 + theta_1 * min_x
    max_y = theta_0 + theta_1 * max_x
    plt.plot([min_x,max_x],[min_y,max_y])
    plt.savefig("resultado.pdf")

def make_data(t0_range,t1_range,X,Y,step=0.1):
    Theta0 = np.arange(t0_range[0],t1_range[1],step)
    Theta1 = np.arange(t1_range[0],t1_range[1],step)
    Theta0,Theta1 = np.meshgrid(Theta0,Theta1)
    Coste = np.empty_like(Theta0)
    #TODO mirar si nos podemos cargar este bucle
    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix,iy] = coste(X,Y, [Theta0[ix, iy],Theta1[ix, iy]])
    return [Theta0,Theta1,Coste]

def visualize_mesh(data):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    #TODO configurar eje
    surface = ax.plot_surface(data[0], data[1] , data[2],cmap=cm.jet,linewidth=0, antialiased=True)
    fig.colorbar(surface,shrink=0.5,aspect=5)
    plt.show()

def visualize_contour(data):
    datos = 0


def main():
    datos = carga_csv('ex1data1.csv')
    # descenso_gradiente(datos,alpha=0.01);
    X = datos[:,:-1]
    Y = datos[:, -1]
    datos_graficos = make_data([-10,10],[-1,4],X,Y)
    visualize_mesh(datos_graficos)


main()
