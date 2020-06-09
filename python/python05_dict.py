# 3. 딕셔너리  중복 X , 추가, 삭제 가능
# {키 : 벨류}
# {key : value}  {인덱스 : 값} <-- 값은 쌍이다

a = {1:'hi', 2:'hello'}
print(a) 
print(a[1])   # hi

b = {'hi' : 1, 'hello' : 2}
print(b['hello'])  # 2

# 딕셔너리 요소 삭제
del a[1]
print(a)  # {2: 'hello'}
del a[2]
print(a)  # {}

a = {1:'a', 1:'b', 1:'c'}
print(a)  # {1: 'c'}  마지막에 덮어쓴 부분이 나온다

b = {1: 'a', 2: 'a', 3: 'a'}
print(b)  # {1: 'a', 2: 'a', 3: 'a'}  키값은 중복 X, 벨류값은 중복 O

a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}
print(a.keys())  # dict_keys(['name', 'phone', 'birth'])
print(a.values())  #  dict_values(['yun', '010', '0511'])
print(type(a))  # <class 'dict'>
print(a.get('name'))  #  yun
print(a['name'])  #  yun
print(a.get('phone')) # 010
print(a['phone'])   # 010

# 조건문 & 반복문 + 구글링 : 언어별로 같은 맥락을 가지고 있다

