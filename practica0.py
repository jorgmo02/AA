import time
import numpy as np
import matplotlib.pyplot as plt

def dot_product(x1, x2):
    """Calcula el producto escalar con un bucle
    y devuelve el tiempo en milisegundos"""
    tic = time.process_time()
    dot = 0
    for i in range (len(x1)):
        dot += x1[i] * x2[i]
    toc = time.process_time()
    return 1000 * (toc - tic)

def fast_dot_product(x1, x2):
    """Calcula el producto escalar vectorizado
    y devuelve el tiempo en milisegundos"""
    tic = time.process_time()
    dot = np.dot(x1, x2)
    toc = time.process_time()
    return 1000 * (toc - tic)

def compara_tiempos():
    sizes = np.linspace(100, 10000000, 20)
    times_dot = []
    times_fast = []
    for size in sizes:
        x1 = np.random.uniform(1, 100, int(size))
        x2 = np.random.uniform(1, 100, int(size))
        times_dot += [dot_product(x1, x2)]
        times_fast += [fast_dot_product(x1, x2)]

    plt.figure()
    plt.scatter(sizes, times_dot, c='red', label='bucle')
    plt.scatter(sizes, times_fast, c='blue', label='vector')
    plt.legend()
    plt.savefig('time.png')


def max_altura(fun, a, b, num_puntos):
    puntos = np.linspace(a, b, num_puntos)
    puntos_func = fun(puntos)
    return np.amax(puntos_func)


def integra_mc(fun, a, b, num_puntos=10000):
    min = 0
    max = max_altura(fun, a, b, num_puntos)

    x = np.random.uniform(a, b, num_puntos)
    y = np.random.uniform(min, max, num_puntos)

    sum = np.sum(y < fun(x))
    res = (b-a) * max * (sum / num_puntos)
    print(res)

    puntos = np.linspace(a, b, num_puntos)
    plt.scatter(x, y, c='red', label='puntos aleatorios')
    plt.plot(puntos, fun(puntos), c='blue', label='function')
    plt.show()

    return res


def my_func(x):
    return (x * 4) + 4


integra_mc(my_func, 0, 8, 200000)