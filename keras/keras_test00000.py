import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

samsung = pd.read_csv('./data/csv/samsung.csv',
                        index_col=0, # None # '일자'가 데이터로 잡힌다.
                        header=0, 
                        sep=',',
                        encoding='CP949')

hite = pd.read_csv('./data/csv/hite.csv',
                        index_col=0, # None
                        header=0,
                        sep=',',
                        encoding='CP949')

print(samsung)

print(samsung.head())

print(hite.head())  # 결측치에 값 넣어야 된다.

print(samsung.shape)  # (700, 1)
print(hite.shape)  # (720, 5)


# None(결측치) 제거 1

samsung = samsung.dropna(axis=0)  # 0 을 넣으면, 행 삭제

print(samsung)  # [509 rows x 1 columns]

print(samsung.shape)  # ( , 1)

hite = hite.fillna(method='bfill')  # bfill = 전날 값으로 채우겠다 
hite = hite.dropna(axis=0)


# None(결측치) 제거 2
# hite = hite[0:509]  
# hite.iloc[0, 1:5] = [10,20,30,40] # iloc = 숫자를 넣는다 / 끝에 값에서 빼기 1 
# hite.loc['2020-06-02', '고가':'거래량'] = ['10','20','30','40'] # loc = "행", "열"

print(hite)  # [509 rows x 5 columns]

# 정렬 바꾸기- 오름차순(ascending)으로 변경하기

samsung = samsung.sort_values(['일자'], ascending=['True'])
hite = hite.sort_values(['일자'], ascending=['True'])

print(samsung)

print(hite)

# 콤마제거, 문자를 상수로 형변환

for i in range(len(samsung.index)):  # '37,000' -> 37000
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace(',',''))

print(samsung)

print(type(samsung.iloc[0,0])) # <class 'int'>

for i in range(len(hite.index)):  # i : 전체 행 / j : 전체 열
    for j in range(len(hite.iloc[i])):
        hite.iloc[i,j] = int(hite.iloc[i,j].replace(',',""))

print(hite)

print(type(hite.iloc[1,1])) # <class 'int'>

print(samsung.shape)  # (509, 1)  509바이 1 (590행 1열)
print(hite.shape)   # (509, 5)  509바이 5 (590행 5열)

samsung = samsung.values
hite = hite.values

print(type(samsung))  # <class 'numpy.ndarray'>
print(type(hite))  # <class 'numpy.ndarray'>

np.save('./data/samsung.npy', arr=samsung)
np.save('./data/hite.npy', arr=hite)

# numpy파일 까지 만들어서, 저장했다.

size = 6 

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):  
        subset = seq[i : (i + size)]      
        aaa.append([item for item in subset])           
    print(type(aaa))
    return np.array(aaa)


# 1.데이터
# npy 불러오기

samsung = np.load('./data/samsung.npy', allow_pickle='True')
hite = np.load('./data/hite.npy', allow_pickle='True')

print(samsung.shape)  # (509, 1)
print(hite.shape)   # (509, 5)

samsung = samsung.reshape(samsung.shape[0], ) # (509, )

samsung = (split_x(samsung, size))
print(samsung.shape)   # (504, 6)  # 6일치씩 잘랐다.

x_sam = samsung[:, 0:5]  # 5일치 자른것까지 x로 하겠다.
y_sam = samsung[:, 5]

print(x_sam.shape)  # (504, 5)
print(y_sam.shape)  # (504, )

x_hit = hite[5:510, :]
print(x_hit.shape)  # (504, )


from sklearn.preprocessing import StandardScaler, MinMaxScaler

# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler_cols = ['시가','고가', '저가', '종가','거래량']
scaler.fit(hite)

# hite= scaler.transform(hite)

# pca = PCA(n_components=1)
# pca.fit(hite)
# hite = pca.transform(hite)

print(hite)

def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)


#2.모델구성

input1 = Input(shape=(5, 1))
input1_1 = LSTM(10, return_sequences= True)(input1)
input1_2 = Dense(20)(input1_1)
input1_3 = Dense(40)(input1_2)
input1_4 = Dropout(0.3)(input1_3)

input2 = Input(shape=(5, 1))
input2_1 = LSTM(20, return_sequences= True)(input2)
input2_2 = Dense(30)(input2_1)
input2_3 = Dense(50)(input2_2)
input2_4 = Dropout(0.5)(input2_3)

merge1 = concatenate([input1_3, input2_3]) 

middle1 = Dense(30)(merge1) 
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)

output1 = Dense(30)(middle1)  
output1_2 = Dense(7)(output1)
output1_3 = Dense(1)(output1_2) 

model = Model(inputs=[input1, input2], 
            outputs = output1_3)

model.summary()  


#3.컴파일, 훈련

model.compile(optimizer='adam', loss='mse')

model.fit([x_sam, x_hit], y_sam, epochs=5)

print(x_sam[-1,:])

print(x_hit[-1, :])