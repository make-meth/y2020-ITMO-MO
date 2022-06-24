from utils import *

def wolfe_search(f, x, p, grad, nabla, max_it = 20):
    '''
    Поиск с условиями Вольфе
    '''
    a = 1
    c1 = 1e-4 
    c2 = 0.9 
    fx = f(*x)

    x_new = x + a * p 
    nabla_new = np.array(grad(*x_new))
    for _ in range(0, max_it):
        if not (f(*x_new) >= fx + (c1*a*np.array(nabla).T @ p) or nabla_new.T @ p <= c2*(np.array(nabla)).T @ p):
            break 
        a *= 0.5
        x_new = x + a * p 
        nabla_new = np.array(grad(*x_new))
    return a

def BFGS(f_batch, batch_size, x0, grad, epochs = 100):
    '''
    Реализация BFGS
    Параметры
    f:      целевая функция 
    x0:     начальная гипотеза
    epochs: максимальное число итераций 
    вывод: 
    x:      найденный минимум 
    '''
    f = f_batch(batch_size)
    
    d = len(x0) # наша размерность 
    # print(f'{x0=}')
    nabla = np.array(grad(*x0)) # градиент в начальной точке
    I = np.eye(d) # единичная матрица

    H = np.copy(I) # начальный обратный гессиан
    x = np.copy(x0)

    points = []
    for i in range(1, epochs):
        if np.linalg.norm(nabla) < 1e-5:
            break 
        print(i)

        p = -H @ nabla # направление поиска
        a = wolfe_search(f, x, p, grad, nabla) # поиск с условиями Вольфе 
        s = np.array([a * p]) # величина шага (dx)
        x_new = x + a * p 
        nabla_new = np.array(grad(*x_new))
        y = np.array([nabla_new - nabla]) # d(nabla f)

        y, s = np.reshape(y,(d,1)), np.reshape(s,(d,1))
        y_trans, s_trans = y.transpose(), s.transpose()
        
        r = 1/(y_trans @ s + 1e-20)

        # это можно вычислить более эффективно без временных матриц        
        li = I-(r*(s @ (y_trans)))
        ri = I-(r*(y @ (s_trans)))
        h_inter = li @ H @ ri
        H = h_inter + (r*(s @ s_trans)) # обновление (обратного) гессиана
        
        nabla = np.copy(nabla_new)
        x = np.copy(x_new)
        
        points.append(x)
    return points

    