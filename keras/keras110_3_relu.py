# ReLU 함수의 값은 입력 값이 0이상이면, 입력 값을 그대로 출력하고 0이하이면, 0을 출력한다. 
# sigmoid와 tanh랑 비교하면, 학습속도가 훨씬 빨라지고 exp함수를 사용하지 않아 
# 연산비용이 크지않고 구현이 매우 간단하다. 
# 그러나 x < 0인 값들에 대해서 기울기가 0으로, 뉴런이 죽을수 있는 단점이 있다.

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)


plt.plot(x, y)
plt.grid()
plt.show()
