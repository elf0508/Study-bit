# 6/3 삼성전자 시가 맞추기- 회귀모델


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

datasets = np.load('./data/data_sam_np_save.npy') 
datasets = np.load('./data/data_hit_np_save.npy') 

print(datasets.shape)  

x = datasets[:, :4]

print(x.shape)        

y = datasets[:, 4:]

print(y.shape)        


# 판다스로 불러들인다.
a = pd.read_csv("./data/csv/data_sam.csv", index_col = 0,
                        header = 0, encoding='cp949', sep =',')

a = a.fillna('0')

a.head()

b = pd.read_csv("./data/csv/data_hit.csv", index_col = 0,
                        header = 0, encoding='cp949', sep =',')

b = a.fillna('0')

b.head()


print(a)  # [720 rows x 6 columns]
print(b)  # [700 rows x 2 columns]

print(a.shape) # (720, 6)
print(b.shape) # (700, 2)

# for i in range(len(a.index)):
#     a.iloc[i,4] = int(a.iloc[i,4].replace(',',''))

# for i in range(len(b.index)):
#     for j in range(len(b.iloc[i])):
#         b.iloc[i,j] = int(b.iloc[i,j].replace(',',''))

# a = a.sort_values(['일자'],ascending=[True])
# b = b.sort_values(['일자'],ascending=[True])

# print(a)
# print(b)


print("====판다스를 넘파이로 바꾸는 함수(.values)========")

a = a.values # 
b = b.values

print(type(a)) # <class 'numpy.ndarray'>
print(type(b)) 

print(a.shape) # (720, 6)
print(b.shape) # (700, 2)


# 전처리 데이터 하기 전에 저장

np.save('./data/csv/data_sam.npy', arr=a) 
np.save('./data/csv/data_hit.npy', arr=b)

# a = np.load('./data/csv/data_sam.npy')
# b = np.load('./data/csv/data_hit.npy')

print("------ a  ------")

print(a)

print("------ b ------")

print(b)

print("------ a.shape  ------")

print(a.shape)

print("------ b.shape  ------")

print(b.shape)

# # df.fillna(0)
# a_0 = a.fillna(0)
# b_0 = b.fillna(0)

# a.head()
# b.head()

# print(a.fillna(0))
# print(b.fillna(0))

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler_cols = ['시가','고가', '저가', '종가','거래량']
scaler.fit(a)
scaler.fit(b)

a = scaler.transform(a)
b = scaler.transform(b)

pca = PCA(n_components=2)
pca.fit(a)
a = pca.transform(a)

pca = PCA(n_components=2)
pca.fit(b)
b = pca.transform(b)

print(a)
print(b)

def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

       
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(a, b, train_size =0.8,random_state= 10)

print("==== x reshape ====")
a = a.reshape(a.shape[0], a.shape[1], 1) 
b = b.reshape(b.shape[0], b.shape[1], 1) 

print("=== a shape ===")
print("a.shape : ", a.shape)
print(a)

print("=== b shape ===")
print("b.shape : ", b.shape)
print(b)


#2. 모델 구성

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM

input1 = Input(shape=(6, 1))
input1_1 = LSTM(10)(input1)
input1_2 = Dense(20)(input1_1)
input1_3 = Dense(40)(input1_2)
input1_4 = Dropout(0.3)(input1_3)

input2 = Input(shape=(6, 1))
input2_1 = LSTM(20)(input2)
input2_2 = Dense(30)(input2_1)
input2_3 = Dense(50)(input2_2)
input2_4 = Dropout(0.5)(input2_3)


####  합병 ######

from keras.layers.merge import concatenate

merge1 = concatenate([input1_3, input2_3]) 

middle1 = Dense(30)(merge1) 
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)


# 병합한 모델을 분리 시킨다 

##### output 모델 구성 ####

output1 = Dense(30)(middle1)  # y_M1의 가장 끝 레이어가 middle1

output1_2 = Dense(7)(output1)

output1_3 = Dense(1)(output1_2)  # y의 열에 맞춰야한다.

### 함수형 모델 명시 ###   Sequential은 맨 위에 명시 해야한다.  model = Sequential()

model = Model(inputs=[input1, input2], 
            outputs = output1_3)

model.summary()  


# callbacks 

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

# earlystopping

es = EarlyStopping(monitor = 'val_loss', patience = 50, verbose =1)

# Tensorboard

ts_board = TensorBoard(log_dir = 'graph', histogram_freq= 0,
                      write_graph = True, write_images=True)

# Checkpoint

modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'

ckecpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                            save_best_only= True)


#3. 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

hist = model.fit(x_train, y_train, epochs =100, batch_size= 64,
                validation_split = 0.2, verbose = 2,
                callbacks = [es,ts_board,ckecpoint])


# 평가, 예측

loss, acc = model.evaluate(x_test, y_test, batch_size = 64)

print('loss: ', loss )

print('acc: ', acc)

y1_pred = model.predict([x_test])

for i in range(5):
    print('종가 : ', y_test[i], '/예측가 : ', y1_pred[i])


# graph

import matplotlib.pyplot as plt

plt.figure(figsize = (10, 5))


# 1

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], c= 'red', marker = '^', label = 'loss')
plt.plot(hist.history['val_loss'], c= 'cyan', marker = '^', label = 'val_loss')
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# 2

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], c= 'red', marker = '^', label = 'acc')
plt.plot(hist.history['val_acc'], c= 'cyan', marker = '^', label = 'val_acc')
plt.title('accuarcy')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
