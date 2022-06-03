from utils import *

def sgd_momentum(batch_size, f, x, *, lr0, epoch, alpha):
    points = np.zeros((epoch, len(x)))
    points[0] = x
    dx = 0
    for i in range(1, epoch):
        dx = alpha * dx - lr0 * np.array(grad_batch(f, batch_size)(*x))
        x = x + dx
        points[i] = x
    return points

def sgd_ada_grad(batch_size, f, x, *, lr0, epoch):
    points = np.zeros((epoch, 2))
    points[0] = x
    s = 0
    for i in range(1, epoch):
        g = np.array(grad_batch(f, batch_size)(*x))
        s += g**2
        x = x - lr0 * (g / np.sqrt(s))
        points[i] = x
    return points

def sgd_rms_prop(batch_size, f, x, *, lr0, epoch, alpha):
    points = np.zeros((epoch, 2))
    points[0] = x
    v = 0
    for i in range(1, epoch):
        g = np.array(grad_batch(f, batch_size)(*x))
        v = alpha * v + (1 - alpha) * g**2
        x = x - lr0 / np.sqrt(v)*g
        points[i] = x
    return points 


def sgd_adam(batch_size, f, x, *, lr0, epoch, alpha, beta):
    points = np.zeros((epoch, 2))
    points[0] = x
    m = 0
    v = 0
    for i in range(1, epoch):
        g = np.array(grad_batch(f, batch_size)(*x))
        m = alpha * m + (1-alpha)*g
        v = beta * v + (1 - beta) * g**2

        m_ = m/(1-alpha)
        v_ = v/(1-beta)

        x = x - lr0*m_/(np.sqrt(v_) + 1e-5)
        points[i] = x
    return points

def sgd_nesterov(batch_size, f, x, *, lr0, epoch, alpha):
    points = np.zeros((epoch, len(x)))
    points[0] = x
    g_im1 = np.zeros(x.size)
    for i in range(1, epoch):
        g_im1 = alpha * g_im1 \
                + lr0 * np.array(grad_batch(f, batch_size)(*(x + alpha * g_im1 )))
        x = x - g_im1
        points[i] = x
    return points   