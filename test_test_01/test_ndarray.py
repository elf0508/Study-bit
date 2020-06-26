# 파이썬의 리스트와 ndarray의 차이 
# ndarray의 슬라이스는 배열의 복사본이 아닌 view이다.
# view란 원래의 데이터를 가리킨다.
# 즉, ndarray의 슬라이스는 원래 ndarray를 변경하게 된다.
# 슬라이스의 복사본 : arr[:].copy() 를 사용한다.

import numpy as np

# 파이썬의 리스트에 슬라이스를 이용한 경우

arr_List = [x for x in range(10)]
print("리스트형의 데이터")   

print("arr_List : ", arr_List)
print()

# 리스트형의 데이터
# arr_List :  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  

print("====================================")

arr_List_copy = arr_List[:]
arr_List_copy[0] = 100

print("리스트의 슬라이스는 복사본이 생성되므로 arr_List에는 arr_List_copy의 변경이 반영되니 않는다.")

print("arr_List : ", arr_List)
print()

# 리스트의 슬라이스는 복사본이 생성되므로 arr_List에는 arr_List_copy의 변경이 반영되지 않는다.  
# arr_List :  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 

print("====================================")

# numpy의 ndarray에 슬라이스를 이용한 경우

arr_numpy = np.arange(10)
print("numpy의 ndarray 데이터")

print("arr_numpy : ", arr_numpy)
print()

# numpy의 ndarray 데이터
# arr_numpy :  [0 1 2 3 4 5 6 7 8 9]

print("====================================")

arr_numpy_view = arr_numpy[:]
arr_numpy_view[0] = 100

print("numpy의 슬라이스는 view(데이터가 저장된 위치의 정보)가 대입되므로 arr_numpy_view를 변경하면 arr_numpy에 반영된다.")
print("arr_numpy : ", arr_numpy)
print()

# numpy의 슬라이스는 view(데이터가 저장된 위치의 정보)가 대입되므로 arr_numpy_view를 변경하면 arr_numpy에 반영된다.
# arr_numpy :  [100   1   2   3   4   5   6   7   8   9]

print("====================================")

# numpy의 ndarray에서 copy()를 사용한 경우

arr_numpy = np.arange(10)
print("arr_numpy에서 copy()를 사용한 경우")

print("arr_numpy : ", arr_numpy)
print()

# arr_numpy에서 copy()를 사용한 경우
# arr_numpy :  [0 1 2 3 4 5 6 7 8 9]

print("====================================")

arr_numpy_copy = arr_numpy[:].copy()
arr_numpy_copy[0] = 100

print("copy()를 사용하면 복사본이 생성되기 때문에 arr_numpy_copy는 arr_numpy에 영향을 미치지 않는다.")
print("arr_numpy : ", arr_numpy)

# copy()를 사용하면 복사본이 생성되기 때문에 arr_numpy_copy는 arr_numpy에 영향을 미치지 않는다.     
# arr_numpy :  [0 1 2 3 4 5 6 7 8 9]






