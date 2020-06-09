# 0~ 9까지있는 분류모델 

from keras.datasets import mnist
from keras.utils import np_utils  #label이 시작하는게 0부터
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print("x_train.shape:",x_train.shape) #(60000,28,28)
print("x_test.shape:",x_test.shape) #(10000,28,28)

print("==================================")

# x_train=x_train.reshape(x_train.shape[0],28,28,1)/255 #0부터 255까지 들어있는 걸--->0부터 1까지로 바꿔줌(minmax)
# x_test=x_test.reshape(x_test.shape[0],28,28,1)/255

x_train=x_train.reshape(x_train.shape[0],28*28)/255
x_test=x_test.reshape(x_test.shape[0],28*28)/255

print(x_train.shape) #(60000,28*28)
print(x_test.shape) #(10000,28*28)
print("=====================================")
y_train=np_utils.to_categorical(y_train) #y는 0부터 시작하기 때문에 np_utils써줘도 된다
y_test=np_utils.to_categorical(y_test)

print(y_train.shape)

print("====================================")

#2. 모델
# gridSearch-->(model, hyperparameter, cv)
# gridSearch의 parameter 의 model을 받기 위해서
# 모델 자체를 진짜 함수로 만든다
# Dense모델 구성
# 모델을 여러번 쓸 수 있다
# gridsearch에 넣기위한 모델(모델에 대한 명시 : 함수로 만듦)
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28, ), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    
    return model
#fit은 gridSearch에서 함

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]  # 5번
    optimizers = ['rmsprop', 'adam', 'adadelta']  # 3번
    dropout = np.linspace(0.1, 0.5, 5)  # 0.1~ 05까지 총 5번
    return{"batch_size": batches, "optimizer":optimizers, "drop": dropout}


#keras 안에 sickit learn 싸는 거 wrapper
#사이킷 런에서 쓸 수 있게 wrapping 한 거
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn = build_model, verbose = 1)

hyperparameters=create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = GridSearchCV(model,hyperparameters, cv=3)  # cv = 3번  총 225번 돌아간다.
search.fit(x_train,y_train)

print(search.best_params_) #params_ = estimators_

'''
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np

#1. data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)                                   # (60000, 28, 28)
print(x_test.shape)                                    # (10000, 28, 28)

x_train = x_train.reshape(x_train.shape[0], 28*28)/225
x_test = x_test.reshape(x_test.shape[0], 28*28)/225

# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)                                    # (60000, 10)

#2. model

# gridsearch에 넣기위한 모델(모델에 대한 명시 : 함수로 만듦)
def build_model(drop=0.5, optimizer = 'adam'):
    inputs = Input(shape= (28*28, ), name = 'input')
    x = Dense(51, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation = 'softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    return model

# parameter
def create_hyperparameters():
    batches = [10, 20, 30, 40, 5]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)                           # 
    return{'batch_size' : batches, 'optimizer': optimizers, 
           'drop': dropout}                                       # dictionary형태

# wrapper
from keras.wrappers.scikit_learn import KerasClassifier # sklearn에서 쓸수 있도로 keras모델 wrapping
model = KerasClassifier(build_fn = build_model, verbose = 1)

hyperparameters = create_hyperparameters()

# gridsearch
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
search = GridSearchCV(model, hyperparameters, cv = 3)            # cv = cross_validation

# fit
search.fit(x_train, y_train)

print(search.best_params_)   # serch.best_estimator_

'''


