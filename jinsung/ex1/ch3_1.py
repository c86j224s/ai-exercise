#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pylab as plt



# 퍼셉트론은 인간이 가중치 설정. 신경망은 가중치 변수를 자동으로 학습하는 능력이 있음.
# 층의 갯수가 3개로, 입력층, 은닉층, 출력층 세개로 구성된 신경망을, 
# 앞서 배운 기준대로, "2층 신경망"이라고 부르자.

# 신경망은 퍼셉트론의 공식에서, h(x)라는 함수를 통해 변환하는 처리가 추가된다.
# 예시:
#   y = h(b + w1 * x1 + w2 * x2)
#   h(a) = 0 if a <= 0
#       or 1 if a > 0
#   (원래 책에는 h(x)라고 되어 있는데, 헷갈려서 a로 바꿔 표기함.)

# h(x)를 활성화 함수(activation function)라고 한다.
# 신경망의 뉴런은 a를 구하는 과정 하나와 a에서 y를 구하는(h를 적용해서) 과정 하나,
# 두단계로 구성된다. 이 두단계가 하나의 뉴런이다.

# 용어 정리. 뉴런 = 노드

# "퍼셉트론에서는 활성화 함수로 계단 함수를 이용한다." : 임계값을 경계로 출력이 휙휙 바뀜..

# 신경망에서 자주 이용하는 활성화 함수 - 시그모이드 함수 (sigmoid function)
#   h(x) = 1 / (1 + exp(-x))


# 우선 계단 함수를 만들어보자.
def p69_a():
    def step_function(x):
        # numpy의 연산자 트릭을 이용하여, 배열도 받을 수 있게 함.
        y = x > 0
        return y.astype(np.int)

    x = np.array([-1.0, 1.0, 2.0])
    print(x)
    y = x > 0
    print(y)
    y = y.astype(np.int)
    print(y)

    # 그래프로도 그려보자
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1) # y 축의 범위 지정
    plt.show()


# 시그모이드 함수를 구현해보자.
# '시그모이드'는 'S자 모양'이란 뜻이다.
def p72_a():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    x = np.arange(-1.0, 1.0, 2.0)
    print(sigmoid(x))

    # 이건 설명하다 말고 numpy의 짱좋은 브로드캐스트 보여주기 위한 막간의...
    t = np.arange(1.0, 2.0, 3.0)
    print(1.0 + t)
    print(1.0 / t)

    # 얘도 그래프로 그려보자
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


# 다른 점 : 
# 시그모이드 함수와 계단 함수의 차이는, 바로 출력의 연속성 혹은 매끄러움이다.
# 기울기와 연속성이 다르다.
# 비슷한 점 : 
# 둘다 입력이 커질 수록 출력이 1 혹은 1에 가까워지는 성질을 가지고 있다.
# 둘 다 비선형 함수이다.
# (선형 함수는 신경망에서 의미가 없는데, 아무리 계층을 깊게 해도 결과에 큰 차이가 없기 때문이다)


# 최근에는 시그모이드 뿐만 아니라, ReLU(Rectified Linear Unit) 함수가 주로 이용되고 있다.
# ReLU는 입력이 0을 넘으면 그 입력을 그대로 출력하고, 0 이하면 0을 출력하는 함수다.
#   h(x) = x if x >= 0
#       or 0 if x <= 0
# 한가지 책에서 설명하지 않는 것이 있는데, 앞서 두 함수들과 달리, y의 최대값이 1이 아니다.
def p77_a():
    def relu(x):
        return np.maximum(0, x)

    x = np.arange(-1.0, 1.0, 2.0)
    print(relu(x))

    # 얘도 그래프로 그려봐야겠지?
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


# 잠깐 또 numpy로 다차원 배열 내적하는 방법 공부하고 넘어간다 --;;
def p77_b():
    A = np.array([1, 2, 3, 4])
    print(A)
    print(np.ndim(A))   # 차원의 수 확인
    print(A.shape)      # 형상 확인
    print(A.shape[0])

    B = np.array([[1, 2], [3, 4], [5, 6]])
    print(B)
    print(np.ndim(B))
    print(B.shape)


# 행렬의 내적=스칼라곱(scalar product)=점곱(dot product)
def p79_a():
    A = np.array([[1, 2], [3, 4]])
    print(A.shape)
    B = np.array([[5, 6], [7, 8]])
    print(B.shape)
    print(np.dot(A, B))

    A = np.array([[1, 2, 3], [4, 5, 6]])
    print(A.shape)
    B = np.array([[1, 2], [3, 4], [5, 6]])
    print(B.shape)
    print(np.dot(A, B))

    C = np.array([[1, 2], [3, 4]])
    print(C.shape)
    print(A.shape)
    try:
        np.dot(A, C)
    except ValueError as e:
        print(f'np.dot(A, C) error: {e}')

    # A의 열 수와 B의 행 수가 일치해야 한다. 그리고 그 결과 형상은 A의 행수와 B의 열 수가 된다.

    # 아래는 B가 2x1의 형상인 것으로 취급된다.
    A = np.array([[1, 2], [3, 4], [5, 6]])
    print(A.shape)
    B = np.array([7, 8])
    print(B.shape)
    print(np.dot(A, B))


# 신경망의 내적
def p82_a():
    # x1, x2 -> (..편향, 활성화 함수 생략..) -> y1, y2, y3 라면..
    # X의 형상은 1x2, W의 형상은 2x3, Y의 형상은 1x3 이다.
    X = np.array([1, 2])
    print(X)
    print(X.shape)
    W = np.array([[1, 3, 5], [2, 4, 6]])
    print(W)
    print(W.shape)
    Y = np.dot(X, W)
    print(Y)
    print(Y.shape)

    # 행렬의 내적은, Y의 갯수에 무관하게 한번의 연산으로 계산하게 해준다는 점에서 중요하다.




def main():
    print('----------')
    p69_a()
    print('----------')
    p72_a()
    print('----------')
    p77_a()
    print('----------')
    p77_b()
    print('----------')
    p79_a()
    print('----------')
    p82_a()


if __name__ == '__main__':
    main()
