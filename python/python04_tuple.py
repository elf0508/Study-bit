# 2. 튜플
# 리스트와 거의 같으나, 객체 안의 값 -->  삭제, 수정(변경)이 안된다.
# 고정값을 넣을때 사용한다.  바뀌지 않는 값
a = (1, 2, 3)
b = 1, 2, 3
print(type(a))  # <class 'tuple'>
print(type(b))  # <class 'tuple'>

# a.remove(2)
# print(a)  # 속성 에러 : 'tuple' object has no attribute 'remove'

print(a + b)  # (1, 2, 3, 1, 2, 3)
print(a * 3)  # (1, 2, 3, 1, 2, 3, 1, 2, 3)

# print(a - 3)  # 속성 에러 : unsupported operand type(s) for -: 'tuple' and 'int'

