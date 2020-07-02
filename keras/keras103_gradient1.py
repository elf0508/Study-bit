import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6
x = np.linspace(-1, 6, 100)  # -1부터 6까지 100간격으로
y = f(x)   # 2차 함수가 나온다.

# 그래프
plt.plot(x, y, 'k-')  # 줄 끗는것
plt.plot(2, 2, 'sk')  # 점 찍는것
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()