import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

plt.rcParams["figure.figsize"] = (30, 15)


def grad_batch(f_batch_size, batch_size):
    def grad_help(*args):
        h = 1e-10
        dim = len(args)
        f = f_batch_size(batch_size)
        return [(
                        f(*[args[j] + (h if j == i else 0) for j in range(dim)])
                        -
                        f(*[args[j] - (h if j == i else 0) for j in range(dim)])
                ) / (2 * h)
                for i in range(dim)]

    return grad_help


def sgd_general(batch_size, f, x, *, lr0, d, epoch):
    points = np.zeros((epoch, len(x)))
    points[0] = x
    for i in range(1, epoch):
        x = x - lr0 * np.exp(-d * i) * np.array(grad_batch(f, batch_size)(*x))
        points[i] = x
    return points


def regression(x, y, batch_size=1, method=sgd_general, **config):
    if config == {}:
        config = {"lr0": 0.5, "d": 0.005, "epoch": 1000}
    x_mat = np.hstack((np.ones((x.shape[0], 1)), x))
    k = x_mat.shape[1]
    batch_choice = lambda batch_size: list(set(np.random.choice(np.arange(x.shape[0]), batch_size, replace=False)))
    f_batch_size = lambda batch_size: \
        lambda *b, batch=batch_choice(batch_size): \
            np.linalg.norm((y[batch] - x_mat[batch].dot(b)))
    bs = method(batch_size, f_batch_size, np.full(k, 1), **config)
    f = f_batch_size(x.shape[0])
    print(f'came close by {f(*bs[-1])}')
    # ax = plt.figure().add_subplot()
    X = np.arange(len(bs))
    # ax.plot(X, np.vectorize(f)(*bs.T))
    # ax.grid()
    if len(x[0]) == 1:
        draw_2d(x, y, bs[-1])
    return bs[-1]


def test(x, y):
    x = x.reshape(len(x))
    ax = plt.figure().add_subplot()
    ax.scatter(x, y)
    ax.grid(True)


def draw_2d(x, y, bs, title=None):
    x = x.reshape(len(x))
    ax = plt.figure().add_subplot()
    if not title is None:
        ax.set_title(title)
    ax.scatter(x, y)
    ax.grid(True)
    tmin = x.min() - 1
    tmax = x.max() + 1
    X = np.array([tmin, tmax])
    Y = (lambda z: bs[0] + bs[1] * z)(X)
    ax.add_line(mlines.Line2D(X, Y, color='green'))


rand = lambda n: (np.random.rand(n) * 2) - 1


def random_perfect_line(n, k, k_limit, ms_limit, debug=False):
    m = rand(n) * ms_limit
    s = rand(n) * ms_limit
    ks = rand(k) * k_limit
    xy = np.array([m + s * k for k in ks])
    x = xy[:, :-1]
    y = xy[:, -1]
    if debug:
        print(f'{m=}\n{s=}\n{x=}\n{y=}')
    return x, y


def test_line(n, k, k_limit, ms_limit, eps):
    x, y = random_perfect_line(n, k, k_limit, ms_limit)
    x += np.array([rand(x.shape[1]) for _ in range(x.shape[0])]) * eps
    y += rand(y.size) * eps
    return x, y


def distance_point(line, point):
    a = np.array(line[:2])
    n = np.array(line[2:])
    return np.linalg.norm((point - a) - (point - a).dot(n) * n)


def distance(line, points):
    return sum(distance_point(line, point) for point in points)


def sgd_general(batch_size, f, x, *, lr0, d, epoch):
    points = np.zeros((epoch, len(x)))
    points[0] = x
    for i in range(1, epoch):
        x = x - lr0*np.exp(-d*i) * np.array(grad_batch(f, batch_size)(*x))
        points[i] = x
    return points


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
        m = alpha * m + (1-alpha) * g
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

class Poly:
    """"
        c_0 + c_1 x + c_2 x^2 + .. + c_deg x^deg
    """
    def __init__(self, *, deg=2, coeffs=None):
        if coeffs is not None:
            self.coeffs = np.array(coeffs)
        else:
            self.coeffs = np.ones(deg+1)

    def __call__(self, x):
        return sum(c_i * x**i for (i, c_i) in enumerate(self.coeffs))

    def grad(self, x):
        return np.array([x**i for (i, c_ip1) in enumerate(self.coeffs[:])]).T[0]

    def hessian(self, x):
        return np.diag(list(x**i for (i, c_ip1) in enumerate(self.coeffs[:])))

#%%

def dogleg_method_step(grad_k, hessian_k, trust_radius):
    hessian_k_inv = np.linalg.inv(hessian_k)
    dx_newton = -np.matmul(hessian_k_inv, grad_k)
    dx_newton_norm = np.linalg.norm(dx_newton)

    if dx_newton_norm <= trust_radius:
        return dx_newton

    dx_steepest = - np.dot(grad_k, grad_k) / np.dot(grad_k, np.dot(hessian_k,grad_k)) * grad_k
    dx_steepest_norm = np.linalg.norm(dx_steepest)

    if dx_steepest_norm >= trust_radius:
        return trust_radius * dx_steepest / dx_steepest_norm

    diff = dx_newton - dx_steepest
    dx_steepest_x_diff = np.matmul(dx_steepest.T, diff)
    discriminant = dx_steepest_x_diff ** 2 - np.linalg.norm(diff) ** 2 * \
                   (np.linalg.norm(dx_steepest) ** 2 - trust_radius ** 2)
    tau = (-dx_steepest_x_diff + np.sqrt(discriminant)) / np.linalg.norm(diff) ** 2
    return dx_steepest + tau * (dx_newton - dx_steepest)

def trust_region_method(func, grad, hessian, x, tr0=1, tr_limit=2 ** 5, epoch=10, eta=0.1):
    x_poly = Poly(coeffs=x)
    points = np.zeros((epoch, len(x)))
    points[0] = x_poly.coeffs
    trust_radius = tr0
    for i in range(1, epoch):
        grad_k = grad(x_poly)
        hessian_k = hessian(x_poly)
        pk = dogleg_method_step(grad_k, hessian_k, trust_radius)

        moved = Poly(coeffs=x_poly.coeffs + pk)

        # Actual reduction.
        act_red = sum(func(x_poly)**2) - sum(func(moved)**2)

        # Predicted reduction.
        # pred_red = -(np.dot(grad_k, pk) + 0.5 * np.dot(pk, np.dot(hessian_k , pk)))
        pred_red = -(np.matmul(grad_k.T, pk) + 0.5 * np.matmul(pk.T, np.dot(hessian_k, pk)))
        # print(f'{pred_red=}\n{act_red=}')
        # print(f'{trust_radius = }')
        # Rho.
        if pred_red == 0.0:
            rhok = 1e99
        else:
            rhok = act_red / pred_red

        # Calculate the Euclidean norm of pk.
        norm_pk = np.linalg.norm(pk)

        # Rho is close to zero or negative, therefore the trust region is shrunk.
        if rhok < 0.25:
            trust_radius = 0.25 * trust_radius
        else:
            # Rho is close to one and pk has reached the boundary of the trust region, therefore the trust region is expanded.
            if rhok > 0.75 and norm_pk == trust_radius:
                trust_radius = min(2.0 * trust_radius, tr_limit)
            else:
                trust_radius = trust_radius

        # Choose the position for the next iteration.
        if rhok > eta:
            x_poly = moved
        else:
            x_poly = x_poly
        points[i] = x_poly.coeffs
    return points

def regression_pdl(x, y, method, **config):
    if config == {}:
        config = {"lr0": 0.5, "d": 0.005, "epoch": 1000}
    f = lambda x_poly: (y - x_poly(x.T[0]))
    jacobi = lambda x_poly: np.array([- x_poly.grad(x[i]) for i in range(len(x))])
    hessian = lambda x_poly: np.matmul(jacobi(x_poly).T, jacobi(x_poly))
    grad = lambda x_poly: 2*np.matmul(jacobi(x_poly).T, f(x_poly))
    bs = method(f, grad, hessian, np.zeros(len(x)), **config)
    # print('hm')
    # print(f'came close by {f(Poly(coeffs=bs[-1]))}')
    return bs[-1]

def test_pdl(coeffs, points, **config):
    coeffs = np.array(coeffs)
    points = np.array(points)
    test_poly = Poly(coeffs=coeffs)
    res = regression_pdl(np.array(points.reshape(-1,1)),test_poly(points),trust_region_method, **config)
    print(f'PDL result for {coeffs} is\n{res}')
    return res


# test_pdl([1, 0, 1], [1, 0, -1], epoch=40, tr0=1)