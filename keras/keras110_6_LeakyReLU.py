# Leakly ReLU 함수의 값은 ReLU와 동일하나 x < 0인 값에 대해서 0.01을 곱하여, 
# ReLU함수의 단점을 보완하였다.

import numpy as np
import matplotlib.pyplot as plt

def LeakyReLU(x):
    return np.maximum(0.01*x, x)

x = np.arange(-5, 5, 0.1)
y = LeakyReLU(x)


plt.plot(x, y)
plt.grid()
plt.show()