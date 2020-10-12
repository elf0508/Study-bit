import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터

a = np.array(range(1, 11))   # a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (10, )
size = 5                    #       0  1  2  3  4  5  6  7  8  9


def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):  # len = length : 길이  i in range(6) : [0, 1, 2, 3, 4, 5]
        subset = seq[i : (i + size)]      # subset = a[0 : 5]   = [1, 2, 3, 4, 5]
        aaa.append([item for item in subset])            # aaa = [[1, 2, 3, 4, 5]]
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("=================================")
print(dataset)

print(a)
x = len(a)

print(x)

aaa = []
for i in range(len(a)- size +1):
    print( 'i :', i)
    subset = a[ i : (i+size)]
    print(subset)
    aaa.append([subset])
    print(aaa)

    '''
    <class 'list'>
=================================
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]

[ 1  2  3  4  5  6  7  8  9 10]
10

i : 0
[1 2 3 4 5]
[[array([1, 2, 3, 4, 5])]]

i : 1
[2 3 4 5 6]
[[array([1, 2, 3, 4, 5])], [array([2, 3, 4, 5, 6])]]

i : 2
[3 4 5 6 7]

i : 3
[4 5 6 7 8]
[[array([1, 2, 3, 4, 5])], [array([2, 3, 4, 5, 6])], [array([3, 4, 5, 6, 7])], [array([4, 5, 6, 7, 8])]]

i : 4
[5 6 7 8 9]
[[array([1, 2, 3, 4, 5])], [array([2, 3, 4, 5, 6])], [array([3, 4, 5, 6, 7])], 
[array([4, 5, 6, 7, 8])], [array([5, 6, 7, 8, 9])]]

i : 5
[ 6  7  8  9 10]
[[array([1, 2, 3, 4, 5])], [array([2, 3, 4, 5, 6])], [array([3, 4, 5, 6, 7])], 
[array([4, 5, 6, 7, 8])], [array([5, 6, 7, 8, 9])], [array([ 6,  7,  8,  9, 10])]] 

    '''