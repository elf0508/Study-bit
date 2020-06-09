# range (class)  : 범위 / 데이터 만들때 사용 : x1, y1 등...

a = range(10)
print(a)  # range(0, 10) 

b = range(1, 11)  # range(1, 11)
print(b)

for i in a:
    print(i)  # 0 ~ 9

print("======================")

for i in b:
    print(i)  # 1 ~ 10

print(type(a)) # <class 'range'>

sum = 0
for i in range(1, 11):
    sum = sum + i
print(sum)  # 55
