#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# numpy 사용 방법..


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread


def ex_a():
    x = np.array([1.0, 2.0, 3.0])
    print(x)
    print(type(x))


def ex_b():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    print(x + y)
    print(x - y)
    print(x * y)     # 원소별 곱셈 (element-wise product)
    print(x / y)


def ex_c():
    x = np.array([1.0, 2.0, 3.0])
    print(x / 2.0)  #  브로드캐스트 (broadcast)


def ex_p38_a():
    A = np.array([[1, 2], [3, 4]])
    print(A)
    print(A.shape)
    print(A.dtype)

    B = np.array([[3, 0], [0, 6]])
    print(B)
    print(A + B)
    print(A * B)    # 이게 왜 이렇게 되는지 난 잘 모르겠다?

    print(A * 10)


def ex_p39():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([10, 20])
    print(A * B)


def ex_p40():
    X = np.array([[51, 55], [14, 19], [0, 4]])
    print(X)
    print(X[0])
    print(X[0][1])

    for row in X:
        print(row)

    X = X.flatten()
    print(X)
    X[np.array([0, 2, 4])]
    print(X > 15)
    print(X[X > 15])


def ex_p42():
    x = np.arange(0, 6, 0.1)
    y = np.sin(x)

    plt.plot(x, y)
    plt.show()


def ex_p43():
    x = np.arange(0, 6, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)

    plt.plot(x, y1, label='sin')
    plt.plot(x, y2, linestyle='--', label='cos')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('sin & cos')
    plt.legend()
    plt.show()


def ex_p44():
    img = imread('lena.png')
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    ex_a()
    ex_b()
    ex_c() 
    ex_p38_a()
    ex_p39()
    ex_p40()
    ex_p42()
    ex_p43()
    ex_p44()

