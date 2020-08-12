# numpy 로 변환

import pymssql as ms
import numpy as np

conn = ms.connect(server = '127.0.0.1', user = 'bit2', password = '1234',
                    database = 'bitdb')

cursor = conn.cursor()

cursor.execute('SELECT * FROM iris2;')

row = cursor.fetchall()

print(row)

conn.close()

print("=========================")
aaa = np.asarray(row)
print(aaa)
print("=========================")
print(aaa.shape)                         # (150, 5)
print("=========================")
print(type(aaa))                         # <class 'numpy.ndarray'>

np.save('./data/test_flask_iris2.npy', aaa)