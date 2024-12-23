import copy
import numpy as np

def kl_gaussian(m1, s1, m2, s2, epsilon=0.00001):
    # kl(custer||prior)

    mean1 = copy.copy(m1)
    std1 = copy.copy(s1)
    mean2 = copy.copy(m2)
    std2 = copy.copy(s2)

    std1 += epsilon
    std2 += epsilon
    a = np.log(std2 / std1)
    zeros_std2 = std2 == 0
    a[zeros_std2] = 0
    b = (std1 ** 2 + (mean1 - mean2) ** 2) / (2 * std2 ** 2)
    return a + b - 1 / 2


def kl_bernoulli(p_value, q_value, epsilon=0.00001):

    p = copy.copy(p_value)
    q = copy.copy(q_value)

    negative_p = p < 0
    negative_q = q < 0
    p[negative_p] = 0
    q[negative_q] = 0
    larger_p = p > 1
    larger_q = q > 1
    p[larger_p] = 1
    q[larger_q] = 1

    zeros_q = q == 0
    q[zeros_q] = epsilon
    ones_q = q == 1
    q[ones_q] = 1 - epsilon

    zeros_p = p == 0
    p[zeros_p] = epsilon
    ones_p = p == 1
    p[ones_p] = 1 - epsilon

    a = p * np.log(p / q)
    b = (1 - p) * np.log((1 - p) / (1 - q))

    zeros_p = p == 0
    a[zeros_p] = 0
    ones_p = p == 1
    b[ones_p] = 0

    return a + b


