# sigmoid (시그모이드) : 0 ~ 1사이의 값
# 함수 값 중심이 0.5로 학습속도가 느리고, exp함수 사용시 연산비용이 많이 들고, 
# 입력값이 일정이상 올라가면 기울기가 거의 0에 수렴하여, gradient vanishing현상이 발생한다.

# 함수 설명 사이트 : https://ratsgo.github.io/deep%20learning/2017/04/22/NNtricks/

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)  # -5 부터 5까지 0.1간격으로
y = sigmoid(x)

print(x.shape, y.shape)

plt.plot(x, y)
plt.grid()   # 격자형태
plt.show()