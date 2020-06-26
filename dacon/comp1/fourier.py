import numpy as np
import matplotlib.pyplot as plt

Fs = 1000
T = 1/Fs
L = 35
t = np.array(range(0,L))*T

X1 = 0.7*np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)
X2 = 0.7*np.sin(2*np.pi*120*t) + np.sin(2*np.pi*170*t)
X3 = 0.7*np.sin(2*np.pi*50*t) + np.sin(2*np.pi*170*t)
X1 = X1 + 2*np.random.randn(len(t))
X2 = X2 + 2*np.random.randn(len(t))
X3 = X3 + 2*np.random.randn(len(t))

plt.subplot(3,3,1)
plt.plot(X1)
plt.subplot(3,3,4)
plt.plot(X2)
plt.subplot(3,3,7)
plt.plot(X3)

Y1 = np.fft.fft(X1)
Y2 = np.fft.fft(X2)
Y3 = np.fft.fft(X3)
P1 = abs(Y1/L)
P2 = abs(Y2/L)
P3 = abs(Y3/L)
P1[2:-1] = 2*P1[2:-1]
P2[2:-1] = 2*P2[2:-1]
P3[2:-1] = 2*P3[2:-1]

f = Fs*np.array(range(0,int(L)))/L
plt.subplot(3,3,2)
plt.plot(f,P1)
plt.subplot(3,3,5)
plt.plot(f,P2)
plt.subplot(3,3,8)
plt.plot(f,P3)


X1 = np.concatenate([X1,X1,X1,X1,X1,X1,X1,X1,X1,X1])
X2 = np.concatenate([X2,X2,X2,X2,X2,X2,X2,X2,X2,X2])
X3 = np.concatenate([X3,X3,X3,X3,X3,X3,X3,X3,X3,X3])

Y1 = np.fft.fft(X1)
Y2 = np.fft.fft(X2)
Y3 = np.fft.fft(X3)
P1 = abs(Y1/L)
P2 = abs(Y2/L)
P3 = abs(Y3/L)
P1[2:-1] = 2*P1[2:-1]
P2[2:-1] = 2*P2[2:-1]
P3[2:-1] = 2*P3[2:-1]
f = Fs*np.array(range(0,int(L*10)))/L

plt.subplot(3,3,3)
plt.plot(f,P1)
plt.subplot(3,3,6)
plt.plot(f,P2)
plt.subplot(3,3,9)
plt.plot(f,P3)

plt.show()
