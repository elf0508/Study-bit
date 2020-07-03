# softmax : 분류
# softmax는  데이터 분포 값을 다 합치면 1이 된다.

import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(1, 5)
y = softmax(x)

ratio = y
labels = y
plt.pie(ratio, labels = labels, shadow = True, startangle=90)
plt.show()
