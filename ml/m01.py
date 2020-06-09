import numpy as np 
import matplotlib.pylab as plt

x = np.arange(0, 10, 0.1)  # arange (0 ~ 10까지 0.1씩 증가시키겠다.)
y = np.sin(x)

plt.plot(x,y)  # x = 0.1씩 증가

plt.show()


