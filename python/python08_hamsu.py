# def로 함수 정의
print("======= 덧셈 ==========")
def sum1(a, b):
    return a + b  # 나온 값을 반환 하겠다
a = 1
b = 2
c = sum1(a, b)

print(c)

print("======= 곱셈 ==========")
# mul1, div1, sub1
def mul1(a , b):
    return a * b
a = 1
b = 2
c = mul1(a , b)
print(c)

print("======= 나눗셈 ==========")
def div1(a, b):
    return a / b
a = 2
b = 4
c = div1(a, b)
print(c)

print("======= 뺄셈 ==========")
def sub1(a, b):
    return a - b
a = 3
b = 1
c = sub1(a, b)
print(c)

print("=====파라미터(매개변수)============")

def sayYeh(): # 변수를 받아들이지 않아도 출력 가능
    return 'Hi'

aaa = sayYeh()
print(aaa)

def sum1(a, b, c):
    return a + b + c  # 나온 값을 반환 하겠다
a = 1
b = 2
c = 34
d = sum1(a, b, c)

print(d)
