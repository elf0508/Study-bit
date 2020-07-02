# lambda : 간략화 된 함수


gradient = lambda x: 2*x - 4

def gradient2(x):  # x 는 input 함수
    temp = 2*x - 4
    return temp

x = 3

print(gradient(x))    # 2

print(gradient2(x))   # 2