# tanh 함수의 값은 -1~1이고, 
# 함수 값 중심이 0으로 학습속도가 개선되었으나, exp함수사용과 입력값이 일정이상 올라가면 
# 기울기가 거의 0의 수렴하는 gradient vanishing 현상은 sigmoid와 동일하다.

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)

plt.plot(x, y)
plt.grid()
plt.show()

