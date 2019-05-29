import numpy as np
import random


def meet_limit_condition(alpha_i, data_i, a, R, C, toler):
    """
    测试alphas[i]是否满足优化条件
    :param alpha_i:alphas[i]
    :param data_i:data_array[i]
    :param a:中心点
    :param R:半径
    :param C:惩罚因子
    :param toler:容忍度
    :return:满足优化条件则返回True，否则返回False
    """
    # if abs(R ** 2 - np.dot((data_i - a), (data_i - a))) > toler and 0 < alpha_i < C:
    Ei = R ** 2 - np.dot((data_i - a), (data_i - a))
    if (Ei < -toler and alpha_i < C) or (Ei > toler and alpha_i > 0):
        return True
    else:
        return False


def selectJrand(i, m):
    """
    随机选择一个整数
    Args:
        i  第一个alpha的下标
        m  所有alpha的数目
    Returns:
        j  返回一个不为i的随机数，在0~m之间的整数值
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def calculate_alpha_j(data_array, alphas, i, j, a):
    """
    data_array: 测试集
    alphas: 旧的alphas值
    i, j: 当前选出的将要进行优化的alpha的下标
    返回值: 新的alphas[j]值
    """

    a1 = np.array(a)
    x1 = np.array(data_array[i])
    x2 = np.array(data_array[j])

    x12 = np.dot(x1, x2)
    x1_2 = np.dot(x1, x1)
    x2_2 = np.dot(x2, x2)

    nu = np.dot(a1, x2) - x2_2 - np.dot(a1, x1) + x1_2 + \
        alphas[i] * (x12 + x1_2) + alphas[j] * (x1_2 - x2_2 + 3 * x12)
    de = 2 * (x1_2 + x2_2 - 2 * x12)

    if de == 0:
        return 0, False

    return -nu / de, True


def calculate_alpha_i(alphas, i):
    """
    alphas: 新的alpha数组
    i: 要更新的alpha值的下标
    返回值： 新的alphas[i]
    """
    alpha_sum = alphas.sum() - alphas[i]
    return 1 - alpha_sum


def smo(train_data, C=0.6, toler=0.001, maxIter=40):
    data_array = np.array(train_data)
    m, n = np.shape(data_array)

    alphas = np.array([1 / m] * m)
    R = 0

    a = np.array([0.0] * n)
    for i in range(m):
        a += alphas[i] * data_array[i]

    iter = 0
    while iter < maxIter:
        changed_flag = 0
        for i in range(m):
            if meet_limit_condition(alphas[i], data_array[i], a, R, C, toler):
                j = selectJrand(i, m)

                L = max(0, alphas[i] + alphas[j] - C)
                H = min(C, alphas[i] + alphas[j])
                if L == H:
                    continue

                new_alpha_j, valid = calculate_alpha_j(
                    data_array, alphas, i, j, a)
                if not valid:
                    continue

                if new_alpha_j < L:
                    new_alpha_j = L
                elif new_alpha_j > H:
                    new_alpha_j = H

                if abs(new_alpha_j - alphas[j]) < 0.001:
                    continue
                else:
                    alphas[j] = new_alpha_j
                    alphas[i] = calculate_alpha_i(alphas, i)
                    changed_flag += 1

                # check_alphas(alphas, C)

                a, R = calculate_a_and_R(data_array, alphas, i, j, C)

        if changed_flag == 0:
            iter += 1
        else:
            iter = 0

    return a, R


def check_alphas(alphas, C):
    """
    检测alphas是否符合要求
    :param alphas:alphas
    :param C:惩罚因子
    :return:符合返回True，否则返回False
    """
    a_sum = 0
    for i in range(alphas.shape[0]):
        if alphas[i] < -0.0001:
            print("alphas" + str(i) + ":" + str(alphas[i]) + " < 0")
        if alphas[i] > C + 0.0001:
            print("alphas" + str(i) + ":" + str(alphas[i]) + " > C")
        a_sum += alphas[i]

    if abs(a_sum - 1) > 0.0001:
        print("alphas sum != 1")
        return False
    else:
        return True


def calculate_a_and_R(data_array, alphas, i, j, C):
    """
    计算a, R
    :param data_array:
    :param alphas:
    :param i:
    :param j:
    :param C:
    :return:
    """
    m, n = np.shape(data_array)
    a = [0] * n
    for l in range(m):
        a += data_array[l] * alphas[l]

    R1 = np.sqrt(np.dot(data_array[i] - a, data_array[i] - a))
    R2 = np.sqrt(np.dot(data_array[j] - a, data_array[j] - a))
    if 0 < alphas[i] < C:
        R = R1
    elif 0 < alphas[j] < C:
        R = R2
    else:
        R = (R1 + R2) / 2.0

    return a, R
