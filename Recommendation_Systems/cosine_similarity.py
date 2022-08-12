import math
import numpy as np
from numpy.linalg import norm


def year_similarity(x, y):
    return math.exp(-(abs(x-y))/10.0)


def cosine_similarity(x, y):
    return np.dot(x, y) / (norm(x) * norm(y))


if __name__ == "__main__":
    # adventure, animation, children, comedy, fiction, fantasy
    m1 = {
        'genre': [1, 1, 1, 1, 1, 0],
        'year': 1995
    }
    m2 = {
        'genre': [1, 0, 1, 0, 0, 1],
        'year': 1995
    }

    print(cosine_similarity(m1['genre'], m2['genre']))
    print(year_similarity(m1['year'], m2['year']))
