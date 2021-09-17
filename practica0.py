import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

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

def compara_tiempos_dot():
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
    tic = time.process_time()
    min = 0
    max = max_altura(fun, a, b, num_puntos)

    x = np.random.uniform(a, b, num_puntos)
    y = np.random.uniform(min, max, num_puntos)

    sum = np.sum(y < fun(x))
    res = (b-a) * max * (sum / num_puntos)
    toc = time.process_time()
    print("-------Operaciones con vectores--------------")
    print("Nuestro resultado {}".format(res))
    print("Resultado de scipy {}".format(scipy.integrate.quad(fun,a,b)))

    puntos = np.linspace(a, b, num_puntos)
    plt.scatter(x, y, c='red', label='puntos aleatorios')
    plt.plot(puntos, fun(puntos), c='blue', label='function')
    # plt.show()

    return res, (toc-tic)


def my_func(x):
    return x**2

def bad_max_altura(fun, a, b, num_puntos):
    x = np.linspace(a, b, num_puntos)
    y = np.empty(x.size)
    for i in range(0, x.size):
        y[i] = fun(x[i])
    return np.amax(y)


def bad_integra_mc(fun, a, b, num_puntos=10000):
    tic = time.process_time()
    min = 0
    max = bad_max_altura(fun, a, b, num_puntos)

    puntos = np.linspace(a, b, num_puntos)
    puntos_func = np.empty(puntos.size)
    for i in range(0, puntos.size):
        puntos_func[i] = fun(puntos[i])

    x = np.random.uniform(a, b, num_puntos)
    y = np.random.uniform(min, max, num_puntos)

    sum = 0
    for i in range(0, x.size):
        if (y[i] < fun(x[i])):
            sum += 1

    res = (b-a) * max * (sum / num_puntos)

    toc = time.process_time()
    print("------------Iterando--------------")
    print("Nuestro resultado {}".format(res))
    print("Resultado de scipy {}".format(scipy.integrate.quad(fun,a,b)))

    plt.scatter(x, y, c='red', label='puntos aleatorios')
    plt.plot(puntos, puntos_func, c='blue', label='function')
    # plt.show()

    return res, (toc-tic)

def compara_tiempos(fun,a,b):
    samples = np.linspace(100, 10000000, 20)
    time_it = []
    time_vec = []
    for sample in samples:
        time_it += [bad_integra_mc(fun,a,b,int(sample))[1]]
        time_vec += [integra_mc(fun,a,b,int(sample))[1]]

    plt.figure()
    plt.scatter(samples, time_it, c='red', label='bucle')
    plt.scatter(samples, time_vec, c='blue', label='vector')
    plt.legend()
    plt.savefig('time_montecarlo.png')


compara_tiempos(my_func, 0, 8)
