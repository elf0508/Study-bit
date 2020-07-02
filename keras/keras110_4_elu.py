# ELU함수는 2015년도에 나온 함수로 ReLU함수에 문제를 해결하고, 
# 출력 값의 중심이 거의 0에 가깝다. 
# 그러나 ReLU와 달리 exp함수를 사용하여 연산비용이 발생한다.

import numpy as np
import matplotlib.pyplot as plt

def elu(x):
    y_list = []
    for x in x:
        if(x>0):
            y = x
        if(x<0):
            y = 0.2*(np.exp(x) -1)
        y_list.append(y)
    return y_list

x = np.arange(-5, 5, 0.1)
y = elu(x)


plt.plot(x, y)
plt.grid()
plt.show()