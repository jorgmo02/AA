from pandas.io.parsers import read_csv
import numpy as np
import matplotlib.pyplot as plt


def carga_csv(filename):
    valores = read_csv(filename,header=None).to_numpy()
    return valores.astype(float)

def punto_en_linea(x,th_0,th_1):
    return th_0 + (x * th_1)

def squared_error(points,theta_0,theta_1):
    m = len(points)
    X = points[:,0]
    Y = points[:,1]
    # Hacer con numpy
    sum_vec = np.sum((punto_en_linea(X,theta_0,theta_1)-Y)**2)
    #for i in range(m):
     #   sum += (punto_en_linea(X[i],theta_0,theta_1)-Y[i])**2
    return sum_vec/(2*m)

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
        print("th0 {} \t\tth1 {} \tsquared error {}".format(theta_0,theta_1,squared_error(datos,theta_0,theta_1)))

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
        print("th0 {} th1 {} squared error {}".format(theta_0,theta_1,squared_error(datos,theta_0,theta_1)))

    plt.plot(X,Y,"x")
    min_x = min(X)
    max_x = max(X)
    min_y = theta_0 + theta_1 * min_x
    max_y = theta_0 + theta_1 * max_x
    plt.plot([min_x,max_x],[min_y,max_y])
    plt.savefig("resultado.pdf")

def main():
    datos = carga_csv('ex1data1.csv')
    descenso_gradiente(datos,alpha=0.01);

main()
