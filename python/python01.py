# 정수형 = int
a = 1
b = 2
c = a + b
print(c)

d = a * b
print(d)

e = a / b
print(e)

# 실수형
a = 1.1
b = 2.2
c = a + b
print(c)   # 3.3000000000000003  /   실수형- 부동소수점(끝의 떨거지 값)에서 오차가 생긴다

d = a * b
print(d)    # 2.4200000000000004

e = a / b
print(e)  # 0.5

# 문자형
a = 'hel'
b = 'lo'
c = a + b
print(c)

# 문자 + 숫자형
a = 123
b = '45'  # ' ' 안에 있으면, 문자가 된다
# c = a + b   # 타입 에러가 나온다
# print(c) 

# 형변환 해보기
# 숫자 --> 문자로 변환 + 문자
a = 123
a = str(a)
print(a)  # 123
b = '45'
c = a + b
print(c)  # 12345

# 문자 --> 숫자로 변환
a = 123
b = '45'
b = int(b)
c = a + b
print(c)   # 168

# 문자열 연산하기 
a = 'abcdefgh'
print(a[0])
print(a[3])
print(a[5])
print(a[-1])
print(a[-2])

print(type(a))  # <class 'str'>  

b = 'xyz'
print(a + b)  # abcdefghxyz

# 문자열 인덱싱 / 시작이 인덱싱은 0 이다
a = 'Hello, Deep learning'
print(a[7])   # D  , (콤마)와 띄어쓰기도 문자다

print(a[-1])   # g
print(a[-2])  # n
print(a[3:9])  # 3 ~ 9까지를 나타낸다 --> lo, De
print(a[3:-5]) # 3 ~ -5 --> 뒤어꺼 빼기 1를 한다 그래서 --> lo, Deep lea 

print(a[:-1])  # Hello, Deep learnin

print(a[1:])  #  ello, Deep learning

print(a[5:-4])  # , Deep lear

