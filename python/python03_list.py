# 자료형
# 1. 리스트 여러가지 자료 사용 가능
# 넘파이는 1가지 자료형만 사용 해야 한다

a = [1,2,3,4,5]
b = [1,2,3,'a','b']
print(b)

print(a[0] + a[3])  # 5
# print(b[0] + b[3])
print(type(a))  # <class 'list'>

print(a[-2])  # 4
print(a[1:3])  #  [2, 3]

a = [1, 2, 3, ['a', 'b', 'c']]
print(a[1])  # 2
print(a[-1])  # ['a', 'b', 'c']
print(a[-1][1])  # b

# 1-2. 리스트 슬라이싱
a = [1,2,3,4,5]
print(a[:2])  # [1, 2]

# 1-3. 리스트 더하기
a = [1, 2, 3]
b = [4, 5, 6]
print(a + b)  # [1, 2, 3, 4, 5, 6]

c = [7, 8, 9, 10]
print(a + c)  # [1, 2, 3, 7, 8, 9, 10]
print( a * 3)  #  [1, 2, 3, 1, 2, 3, 1, 2, 3]

# print(a[2] + 'hi') 
print(str(a[2]) + 'hi')   # 3hi

f = '5'
# print(a[2] + f)
print(a[2] + int(f))  # 8

# 리스트 관련 함수
a = [1, 2, 3]
a.append(4)
print(a)  # [1, 2, 3, 4]  / .append : 덧붙이다 / 리스트에 인자를 추가할 때, 사용한다

# a = a.append(5)  # 오류 : None 이라고 뜬다

a.append(5)
print(a)  # [1, 2, 3, 4, 5]

a = [1, 2, 3, 4]
a.sort()  # .sort : 정렬하겠다
print(a)  # [1, 2, 3, 4]

a.reverse()  # .reverse : 역순
print(a)   # [4, 3, 2, 1]

print(a.index(3))  # == a[3]과 같은 뜻 이다
print(a.index(1))   # == a[1]과 같은 뜻 이다

a.insert(0, 7)  # .insert : 삽입하다
print(a)   # [7, 4, 3, 2, 1]
a.insert(3, 3)
print(a)  #  [7, 4, 3, 3, 2, 1]

a.remove(7)  # .remove : 없애다
print(a)  # [4, 3, 3, 2, 1]

a.remove(3)
print(a)  # [4, 3, 2, 1]   먼저 걸린 숫자가 지워진다