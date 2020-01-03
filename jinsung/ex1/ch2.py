#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


# 퍼셉트론(perceptron) : Frank Resenblatt가 1957년에 고안한 알고리즘.
#   다수의 입력 신호를 받아 하나의 출력 신호를 내보냄.
#   x (입력신호), y (출력신호), w(가중치), theta(임계값)
#   입력신호에 가중치를 곱하여 합산한 결과가 가중치를 넘을 때만 출력신호가 1이 됨. 아니면 0.
#   가중치와 임계값의 조절로 AND 게이트, NAND 게이트, OR 게이트 등을 만들 수 있음.
#   (학습이란 이런 변수의 조절을 사람이 하는 것이 아닌, 기계가 자동으로 하게 하는 것)
#   예시 : 
#   y = 0 if w1 * x1 + w2 * x2 <= theta
#    or 1 if w1 * x1 + w2 * x2 > theta


# 단순 AND 게이트 구현.
def p51_a():
    def AND(x1, x2):
        w1, w2, theta = 0.5, 0.5, 0.7
        tmp = x1*w1 + x2*w2
        if tmp <= theta:
            return 0
        elif tmp > theta:
            return 1

    print(AND(0, 0))
    print(AND(1, 0))
    print(AND(0, 1))
    print(AND(1, 1))


# 가중치와 편향 도입
# theta를 -b로 치환하여 좌항으로 옮기면, 다음과 같은 식이 됨.
# 변형된 식 : 
#   y = 0 if b + w1 * x1 + w2 * x2 <= 0
#    or 1 if b + w1 * x1 + w2 * x2 > 0
# 여기서 b를 편향(bias)라고 함.
def p52_a():
    x = np.array([0, 1])
    w = np.array([0.5, 0.5])
    b = -0.7
    print(w * x)
    print(np.sum(w * x))
    print(np.sum(w * x) + b)


# 다음은 가중치와 편향을 도입해 (+numpy), AND 게이트를 구현한 것.
def p53_a():
    def AND(x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.7
        tmp = np.sum(w*x) + b
        if tmp <= 0:
            return 0
        elif tmp > 0:
            return 1

    print(AND(0, 0))
    print(AND(1, 0))
    print(AND(0, 1))
    print(AND(1, 1))


# NAND 게이트와 OR 게이트도 구현해보자.
def p53_b():
    def NAND(x1, x2):
        x = np.array([x1, x2])
        w = np.array([-0.5, -0.5])
        b = 0.7
        tmp = np.sum(w*x) + b
        if tmp <= 0:
            return 0
        elif tmp > 0:
            return 1   

    def OR(x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.2
        tmp = np.sum(w*x) + b
        if tmp <= 0:
            return 0
        elif tmp > 0:
            return 1   

    print(NAND(0, 0))
    print(NAND(1, 0))
    print(NAND(0, 1))
    print(NAND(1, 1))

    print(OR(0, 0))
    print(OR(1, 0))
    print(OR(0, 1))
    print(OR(1, 1))


# 하나의 퍼셉트론은 직선(선형)만 표현할 수 있지만,
# 다층 퍼셉트론(multi-layer perceptron)은 곡선(비선형)을 표현할 수 있다. 
# XOR 게이트를 만들기 위해서는 AND, NAND, OR 게이트를 조합하면 된다.
def p59_a():
    # p59의 코드를 그대로 치기 귀찮아서...
    def generate_perceptron(w1_, w2_, b_):
        def perceptron(x1, x2):
            x = np.array([x1, x2])
            w = np.array([w1_, w2_])
            b = b_
            tmp = np.sum(w*x) + b
            if tmp <= 0:
                return 0
            elif tmp > 0:
                return 1              

        return perceptron

    AND = generate_perceptron(0.5, 0.5, -0.7)
    OR = generate_perceptron(0.5, 0.5, -0.2)
    NAND = generate_perceptron(-0.5, -0.5, 0.7)

    def XOR(x1, x2):
        s1 = NAND(x1, x2)
        s2 = OR(x1, x2)
        y = AND(s1, s2)
        return s1, s2, y

    print(XOR(0, 0))
    print(XOR(1, 0))
    print(XOR(0, 1))
    print(XOR(1, 1))

    # XOR 게이트의 퍼셉트론 네트워크를 그려보면, 
    # x1, x2의 0층, s1, s2의 1층, y의 2층으로 총 3개 층으로 구성된다.
    # 그런데, 사실 가중치를 갖는 층은 0->1층, 1->2층이므로,
    # 가중치 계산이 들어가는 층을 기준 삼아 "2층 퍼셉트론"으로 칭하도록 하자.
    

# 단층 퍼셉트론으로 구현하지 못하는 것을 층을 늘려 다층 퍼셉트론으로 구현 가능하다.


# NAND 게이트의 조합으로 컴퓨터를 만들 수 있다.
# (NAND 게이트를 퍼셉트론으로 만들 수 있으니, 퍼셉트론으로 컴퓨터를 만들 수도 있다.)
# 참고서적 : <The Elements of Computing Systems: Building a Modern Computer from First Principles>(The MIT Press, 2005)


def main():
    p51_a()
    print('----------')
    p52_a()
    print('----------')
    p53_a()
    print('----------')
    p53_b()
    print('----------')
    p59_a()


if __name__ == '__main__':
    main()
