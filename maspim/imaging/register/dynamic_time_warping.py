import numpy as np


def dtw(x, y, q=2, d=None):
    """
    modified from https://rtavenar.github.io/blog/softdtw.html
    """
    if d is None:
        d = lambda a, b: abs(a - b)

    inf = float('inf')

    n = x.shape[0]
    m = y.shape[0]
    R = np.empty((n, m), dtype=float)

    for i in range(len(x)):
        for j in range(len(y)):
            R[i, j] = d(x[i], y[j]) ** q
            if i > 0 or j > 0:
                R[i, j] += min(
                R[i-1, j-1] if (i > 0 and j > 0) else inf,
                R[i-1, j  ] if i > 0             else inf,
                R[i  , j-1] if j > 0             else inf
                # Note that these 3 terms cannot all be
                # inf if we have (i > 0 or j > 0)
                )

    return R


def path_from_matrix(arr):
    """Determine optimal path through backtracking"""
    n, m = arr.shape
    I = [n - 1]
    J = [m - 1]
    while (I[-1] > 0) or (J[-1] > 0):
        i = I[-1]
        j = J[-1]
        if i == 0:  # only possibility is to go right
            I.append(i)
            J.append(j - 1)
        elif j == 0:  # only possibility is to up
            I.append(i - 1)
            J.append(j)
        else:  # determine best direction as the one with the smallest value
            vals = [  # put diag first for highest priority
                arr[i - 1, j - 1],  # left and up
                arr[i, j - 1],  # left
                arr[i - 1, j]  # up
            ]
            direction = np.argmin(vals)
            if direction == 1:  # left
                I.append(i)
                J.append(j - 1)
            elif direction == 2:  # up
                I.append(i - 1)
                J.append(j)
            else:
                I.append(i - 1)
                J.append(j - 1)
    return I[::-1], J[::-1]


def transform_series_according_to_path(x, y, path):
    return x[path[0]], y[path[1]]


def dtw_distance(s, t, w: int):
    """
    Compute Dynamic Time Warping with distance constraint.

    https://en.wikipedia.org/wiki/Dynamic_time_warping
    """
    n = s.shape[0]
    m = t.shape[0]

    # adjust window size
    w  = max(w, abs(n - m))

    _dtw = np.full((n, m), np.inf)
    _dtw[0, 0] = 0

    for i in range(n):
        for j in range(max(0, i - w), min(m, i + w)):
            cost = abs(s[i] - t[j])
            _dtw[i, j] = cost + min(_dtw[i - 1, j],  # insertion
                                    _dtw[i, j - 1],  # deletion
                                    _dtw[i - 1, j - 1])  # match
    return _dtw


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n = 15
    m = 120
    x = np.linspace(0, 2 * np.pi, n)
    s = np.sin(x)
    x_distort = np.cumsum(np.random.random(m))
    x_distort *= 2 * np.pi / x_distort.max()
    t = np.sin(x_distort)

    u = np.zeros(n + 5)
    u[2:2+n] = s
    v = np.zeros(n + 5 + 10)
    v[5:5 + n] = s

    # u = s
    # v = t

    # plt.figure()
    # plt.plot(s)
    # plt.plot(t)
    # plt.show()
    #
    # plt.figure()
    # plt.plot(x)
    # plt.plot(x_distort)
    # plt.show()

    arr = dtw(u, v)

    path = path_from_matrix(arr)

    plt.figure()
    plt.plot(u, c='C0')
    plt.scatter(range(len(u)), u, c='C0')
    plt.plot(v,c='C1')
    plt.scatter(range(len(v)), v,c='C1')
    plt.show()

    plt.figure()
    plt.imshow(arr)
    plt.plot(path[1], path[0], c='white')
    plt.scatter(path[1], path[0], c='white')
    plt.show()


    ut, vt = transform_series_according_to_path(u, v, path)

    plt.figure()
    plt.plot(ut)
    plt.plot(vt)
    plt.show()



