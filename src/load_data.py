import numpy as np


def load_train_data(path, minIndex, maxIndex, dim=4):
    """

    :param path:
    :param minIndex:
    :param maxIndex:
    :return:
    """
    data = []

    i = -1
    with open(path) as f:
        for line in f.readlines():  # get data by lines
            if line == "\n":  # empty line
                break
            i += 1
            if minIndex <= i < maxIndex:
                feature = line.split(',')
                feature.pop()  # abandon last element, because it is label
                data.append(list(map(float, feature)))  # append to data

    return np.array(data)[:, 0:dim]


def load_test_data(path, type_num=0, minIndex=0, maxIndex=150, dim=4):
    """
    :param path:数据集路径
    :param type_num: 训练集的类型，范围[0, 2]
    :param minIndex: 测试集的范围[minIndex, maxIndex)
    :param maxIndex: 测试集的范围[minIndex, maxIndex)
    :return: 测试集，标签
    """
    data = []
    label = []

    i = -1
    with open(path) as f:
        for line in f.readlines():  # get data by lines
            if line == "\n":  # empty line
                break
            i += 1
            if minIndex <= i < maxIndex:
                feature = line.split(',')
                feature.pop()  # abandon last element, because it is label
                data.append(list(map(float, feature)))  # append to data

                if type_num * 50 <= i < (type_num + 1) * 50:
                    label.append(1)
                else:
                    label.append(0)

    return np.array(data)[:, 0:dim], label
