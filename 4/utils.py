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
    ax = plt.figure().add_subplot()
    X = np.arange(len(bs))
    ax.plot(X, np.vectorize(f)(*bs.T))
    ax.grid()
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


hm = lambda: print("hm")
