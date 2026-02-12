import numpy as np


def dtw(s, t):
    """
    Compute Dynamic Time Warping.

    https://en.wikipedia.org/wiki/Dynamic_time_warping
    """
    n = s.shape[0]
    m = t.shape[0]
    _dtw = np.full((n, n), np.inf)
    _dtw[0, 0] = 0

    for i in range(n):
        for j in range(m):
            cost = abs(s[i] - t[j])
            _dtw[i, j] = cost + min(_dtw[i - 1, j],  # insertion
                                    _dtw[i, j - 1],  # deletion
                                    _dtw[i - 1, j - 1])  # match
    return _dtw


def dtw_distance(s, t, w: int):
    """
    Compute Dynamic Time Warping with distance constraint.

    https://en.wikipedia.org/wiki/Dynamic_time_warping
    """
    n = s.shape[0]
    m = t.shape[0]

    # adjust window size
    w  = max(w, abs(n - m))

    _dtw = np.full((n, n), np.inf)
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

    n = 100
    m = 120
    x = np.linspace(0, 2 * np.pi, n)
    s = np.sin(x)
    x_distort = np.cumsum(np.random.random(m))
    x_distort *= 2 * np.pi / x_distort.max()
    t = np.sin(x_distort)

    plt.figure()
    plt.plot(s)
    plt.plot(t)
    plt.show()

    plt.figure()
    plt.plot(x)
    plt.plot(x_distort)
    plt.show()

    arr = dtw(s, t)
