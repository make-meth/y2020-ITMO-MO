import numpy as np
from matplotlib import pyplot as plt
# Обобщенная функция для вычисления градиента.
def grad(f): 
    def grad_help(*args):
        h = 1e-5
        dim = len(args)
        return [(f(*[args[j] + (h if j == i else 0) for j in range(dim)]) -
                f(*[args[j] - (h if j == i else 0) for j in range(dim)]))/(2*h)
                for i in range(dim)]
    return grad_help

# Градиентный спуск
# Число вычислений grad f  = epoch - 1
# Число вычислений f = 0
def gradient_descent(f, lr0, d, epoch, x, counters = [0,0]):
    points = np.zeros((epoch, 2))
    points[0] = x
    for i in range(1, epoch):
        x = x - lr0*np.exp(-d*i) * np.array(grad(f)(*x))
        points[i] = x  
    
    counters[1] = epoch -1
    return points

# Градиентный спуск с постоянным шагом.
def gradient_descent_const(f, lr, epoch, x):
    return gradient_descent(f, lr, 0, epoch, x)

# Метод золотого сечения
# a,b -- границы промежутка, внутри которого осуществляется поиск
# eps -- точность

def golden(f, a, b, eps, points_check = [], counters = [0,0]):
    points_check = []
    phi = (1 + np.sqrt(5))/2
    def min_rec(f, eps, a, b, fx1, fx2):
        if b-a < eps:
            return (a+b)/2
        else:
            t = (b-a)/phi
            x1, x2 = b - t, a + t

            if fx1 == None:
                fx1 = f(x1)
                points_check.append(x1)
                counters[0] += 1
            if fx2 == None:
                fx2 = f(x2)
                points_check.append(x2)
                counters[0] += 1
            
            if fx1 >= fx2:
                return min_rec(f, eps, x1, b, fx2, None)
            else:
                return min_rec(f, eps, a, x2, None, fx1)
    return min_rec(f, eps, min(a,b), max(a,b), None, None) 

# Градиентный спуск на основе метода золотого сечения
# Число вычислений f -- epoch - 1.
def gradient_golden_sec(f, lr, epoch, x, eps, counters = [0,0]):
    points = np.zeros((epoch, 2))
    points[0] = x
    for i in range(1, epoch):
        d = np.array(grad(f)(*x))
        f_line = lambda a: f((x - a*d)[0], (x - a*d)[1])
        x = x - golden(f_line, 0, lr, eps, counters) * d
        points[i] = x
    counters[1] = epoch - 1 
    return points

# Отрисовка
def method(f, tmin, tmax, lr0, epoch = 200, d = 0, x = [0,0]):
    points = gradient_descent(f, lr0, d, epoch, x)
    t = np.linspace(tmin, tmax, 100)
    X, Y = np.meshgrid(t, t)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(f(points[:, 0], points[:, 1]))
    ax1.grid()
    ax2.plot(points[:, 0], points[:, 1], 'o-')
    ax2.contour(X, Y, f(X, Y), levels=sorted(list(set([f(*p) for p in points] + list(np.linspace(-1, 1, 100))))))
    print(f(points[-1, 0], points[-1, 1]))   