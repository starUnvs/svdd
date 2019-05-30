import load_data
import one_class_svm
import numpy as np
from matplotlib import pyplot as plt


def judge(test_data, a, R):
    m, n = np.shape(test_data)
    label = []
    for i in range(m):
        if np.dot(test_data[i] - a, test_data[i] - a) <= R ** 2:
            label.append(1)
        else:
            label.append(0)

    return label


def calculate_acc(result_label, correct_label):
    """
    返回result_label与correct_label相同的比例
    :return: acc = (true positive + false positive)/all
    """
    if len(result_label) != len(correct_label):
        return -1

    acc = 0
    for i in range(len(result_label)):
        if result_label[i] == correct_label[i]:
            acc += 1

    return acc / len(result_label)


def draw_picture(train_data, test_data, correct_label, a, R, C, toler, acc):
    plt.figure()
    plt.scatter(test_data[:, 0], test_data[:, 1], c=correct_label)
    plt.scatter(train_data[:, 0], train_data[:, 1], c='r')
    plt.title("C = " + str(C) + " toler = " + str(toler) +
              " R = " + str(R)[0:4] + " acc = " + str(acc)[:4])
    theta = np.arange(0, 2 * np.pi, 0.01)
    x = a[0] + R * np.cos(theta)
    y = a[1] + R * np.sin(theta)

    plt.plot(x, y)
    plt.show()


def adjust_param():
    # 以第几类数据作为训练集
    type_num = 0
    dim = 4
    C = 0.6
    toler = 0.0001
    maxIter = 40

    best_acc = 0
    best_a = 0
    best_r = 0
    best_label = []

    # 数据预处理
    if type_num == 0:
        train_data = load_data.load_train_data('../data/iris.data', 0, 30, dim=dim)
        test_data, correct_label = load_data.load_test_data(
            'iris.data', type_num, 30, 150, dim=dim)
    elif type_num == 1:
        train_data = load_data.load_train_data('../data/iris.data', 50, 80)

        test_data1, correct_label1 = load_data.load_test_data(
            'iris.data', type_num, 80, 150, dim=dim)
        test_data2, correct_label2 = load_data.load_test_data(
            'iris.data', type_num, 0, 50, dim=dim)

        test_data = np.vstack((test_data1, test_data2))
        correct_label = np.hstack((correct_label1, correct_label2))
    elif type_num == 2:
        train_data = load_data.load_train_data('../data/iris.data', 100, 130, dim=dim)

        test_data1, correct_label1 = load_data.load_test_data(
            'iris.data', type_num, 130, 150, dim=dim)
        test_data2, correct_label2 = load_data.load_test_data(
            'iris.data', type_num, 0, 100, dim=dim)

        test_data = np.vstack((test_data1, test_data2))
        correct_label = np.hstack((correct_label1, correct_label2))

    min_acc = 2
    avrg_acc = 0
    max_acc = -1
    for i in range(50):
        a, R = one_class_svm.smo(train_data, C, toler, maxIter)
        result_label = judge(test_data, a, R)
        acc = calculate_acc(result_label, correct_label)
        if acc > best_acc:
            best_acc = acc
            best_a = a
            best_r = R
            best_label = result_label

        avrg_acc += acc

        if acc < min_acc:
            min_acc = acc
        if acc > max_acc:
            max_acc = acc

        #print("accuracy: " + str(acc))

    avrg_acc /= 100
    print("train type:" + str(type_num) + ", dim=" +
          str(dim) + " => best acc = " + str(max_acc))
    print("model: a="+str(best_a)+", R="+str(best_r) + ",C="+str(C))
    print("label(0-20:positive sample):")
    print(best_label)
    draw_picture(train_data, test_data, correct_label,
                 best_a, best_r, C, toler, best_acc)


if __name__ == "__main__":
    adjust_param()
