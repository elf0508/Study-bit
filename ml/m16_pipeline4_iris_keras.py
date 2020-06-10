# iris를 케라스 파이프라인 구성
# 당연히 RandomizedSearchCV 구성
# keras 98 참조할것

# 97번을 RandomizedSearchCV로 변경

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense, MaxPool2D

# 1.데이터
dataset = load_iris()
x = dataset.data   
y = dataset.target 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=43)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(x_train.shape)
print(x_test.shape)  
print(y_train.shape) 
print(y_test.shape)

# 2.모델
def build_model(optimizer='adam') :
    input = Input(shape=(4,))
    x = Dense(512, activation='relu')(input)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model
    
def create_hyperparameters() :
    
    batches = [512, 256, 128]  # 이거말고도 epo, node, activation 다 넣을 수 잇음...
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1,0.5, 5)  
    return {'model__batch_size': batches, "model__optimizer": optimizers}

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor 
model = KerasClassifier(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()

from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pipe = Pipeline([("scaler",StandardScaler()),('model', model)]) 

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(pipe, hyperparameters, cv=3) 

# 그리드서치 자체를 fit 때려버리네?
# 3. 훈련
search.fit(x_train, y_train)

# 4. 평가,예측
acc = search.score(x_test, y_test)

print('best_params_은', search.best_params_)
print('sore은 ',acc)

'''
# iris를 케라스 파이프라인 구성
# 당연히 RandomizedSearchCV 구성
from sklearn.datasets import load_iris
import numpy as np
from keras.utils import np_utils
from keras.layers import Dense, Input, Dropout
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from keras.models import Model
from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2)


iris=load_iris()
x = iris.data
y = iris.target
y = np_utils.to_categorical(y)

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)
print("x_train.shape:",x_train.shape) #(120, 4)
print("x_test.shape:",x_test.shape)   #(30, 4)

def build_model(drop=0.3, optimizer='adam',act='relu'):
    drop=0.1
    inputs = Input(shape=(4,))
    x = Dense(512,activation=act,name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256,activation=act,name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation=act,name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation='softmax',name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model. compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['acc'])

    return model

def create_hyperparameters(): #epoch, activation...다 넣을 수 있다
    batches = [128, 256, 512] #batch 종류
    optimizers = ['rmsprop', 'adam', 'adadelta'] #optimizer의 종류
    dropout = np.linspace(0.1,0.5,5).tolist() #0.1부터 0.5까지 5등분 #그냥 linspace 쓰면 오류가 난다. # numpy오류
    epochs = [30,50,100]
    activation = ['relu','elu',leaky]
    
    # model 넣는게 아님/ model을 정의해준 kerasclassifier 넣어주어야 함
    return{"kerasclassifier__batch_size": batches, "kerasclassifier__optimizer" : optimizers, "kerasclassifier__drop" : dropout, "kerasclassifier__epochs": epochs,
                                                                                 "kerasclassifier__act": activation}

# dense모델은 keras
# make_pipeline 과 pipeline 차이
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn = build_model, verbose=1)
hyperparameters=create_hyperparameters()

kfold=KFold(n_splits=5, shuffle=True)

# pipe = Pipeline([("scaler",StandardScaler()),("model",model)])

pipe = make_pipeline(StandardScaler(),model)

search = RandomizedSearchCV(pipe, hyperparameters, cv=kfold)
search.fit(x_train, y_train)

acc = search.score(x_test, y_test)

print("acc:", acc)
print("--------------------")
print(search.best_params_)
print("----------------------")
print(search.best_estimator_)
'''