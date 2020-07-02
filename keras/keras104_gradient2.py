import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6

gradient = lambda x: 2*x - 4  
# f함수 미분 / f = x*2 - 4x + 6 --> 2x - 4 * 1 + 6 * 0 --> 2x - 4 (2차 함수)
# 최저의 f 를 찾는 기울기가 0가 되는 지점을 찾는다.

x0 = 0.0
# MaxIter = 10    # <-- 숫자 바꿔보기
# learning_rate = 0.25  # <-- 숫자 바꿔보기 / 케라스에서는 : optimizer / 3차 함수에서는 숫자 크게

# MaxIter = 10    
# learning_rate = 0.5     

# MaxIter = 5   
# learning_rate = 0.5   

MaxIter = 10   
learning_rate = 0.15 

print("step\tx\tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))

# step    x       f(x)
# 00      0.00000    6.0

# 경사하각법 기본 예제

for i in range(MaxIter):  # 10번 돌릴거다
    x1 = x0 - learning_rate * gradient(x0)
    x0 = x1

    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0)))

'''
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
# plt.show()

gradient = lambda x: 2*x - 4  
# f함수 미분 / f = x*2 - 4x + 6 --> 2x - 4 * 1 + 6 * 0 --> 2x - 4 
# 최저의 f 를 찾는 기울기가 0가 되는 지점을 찾는다.

x0 = 0.0
MaxIter = 10
learning_rate = 0.25

print("step\tx\tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))

# step    x       f(x)
# 00      0.00000    6.0

# 경사하각법 기본 예제

for i in range(MaxIter):  # 10번 돌릴거다
    x1 = x0 - learning_rate * gradient(x0)
    x0 = x1

    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0)))


# step    x       f(x)
# 00      0.00000 6.00000
# 01      1.00000 3.00000
# 02      1.50000 2.25000
# 03      1.75000 2.06250
# 04      1.87500 2.01562
# 05      1.93750 2.00391
# 06      1.96875 2.00098
# 07      1.98438 2.00024
# 08      1.99219 2.00006
# 09      1.99609 2.00002
# 10      1.99805 2.00000

'''









