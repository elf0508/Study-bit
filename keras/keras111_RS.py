# 107번을 Activation 넣어서 완성하시오.

########################

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, MaxPooling2D, Dense, LSTM
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam
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

from keras.layers import LeakyReLU

leaky = LeakyReLU(alpha = 0.2)

# gridsearch에 넣기위한 모델(모델에 대한 명시 : 함수로 만듦)
def build_model(drop=0.5, optimizer = Adam, lr = 0.01, act = 'relu'):
    if act == 'leaky':
        act = leaky

    inputs = Input(shape= (28*28, ), name = 'input')

    x = Dense(51, activation = 'act', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)

    x = Dense(256, activation = 'act', name = 'hidden2')(x)
    x = Dropout(drop)(x)

    x = Dense(128, activation = 'act', name = 'hidden3')(x)
    x = Dropout(drop)(x)

    outputs = Dense(10, activation = 'softmax', name = 'output')(x)

    model = Model(inputs = inputs, outputs = outputs)

    opt = optimizer(lr = lr)

    model.compile(optimizer = opt, metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    return model

# parameter
def create_hyperparameters(): # epochs, node, acivation 추가 가능
    batches = [384, 256]
    optimizers = [Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam]
    learning_rate = [0.05, 0.1, 0.25]
    dropout = np.linspace(0.3, 0.5, 3).tolist()   
    activation = ['tanh', 'relu', 'leaky', 'elu', 'selu']                      
    return {'batch_size' : batches, 'optimizer': optimizers, 'lr':learning_rate,
           'drop': dropout}                                       

# wrapper
from keras.wrappers.scikit_learn import KerasClassifier          
model = KerasClassifier(build_fn = build_model, verbose = 1, epochs = 1)

hyperparameters = create_hyperparameters()

# gridsearch
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
# search = RandomizedSearchCV(model, hyperparameters, cv = 3, n_jobs = 5)          
search = RandomizedSearchCV(model, hyperparameters, cv = 3, n_iter = 5)                        

# fit
search.fit(x_train, y_train)

print(search.best_params_)  


score = search.score(x_test, y_test)
print('acc: ', score)

# 터짐

'''
# 107번을 activation 넣어서 완성하시오

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM
from keras.layers import Conv2D, Flatten, Activation
from keras.layers import MaxPooling2D, Dropout
from keras.wrappers.scikit_learn import KerasClassifier    
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.optimizers import Adam, RMSprop, SGD, Adadelta,Adagrad, Nadam, Adamax
from keras.activations import relu,selu,elu,sigmoid,softmax
# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = np_utils.to_categorical(y_train)      # label이 0부터 시작함
y_test = np_utils.to_categorical(y_test)        # label이 0부터 시작함
print(y_train.shape)                # (60000, 10)
print(y_test.shape)                 # (10000, 10)


# 2. 모델링
def build_model(drop, activation, optimizer, lr):
    inputs = Input(shape = (28, 28), name = 'input')
    x = LSTM(64, return_sequences = True, activation=activation, name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = LSTM(32,activation=activation, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(16, activation=activation, name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation=activation, name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer(lr=lr), metrics = ['accuracy'],
                  loss = 'categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [10, 20, 30, 40, 50]
    activation = [relu,elu,sigmoid,softmax]
    lr = [0.1, 0.01, 0.001, 0.005, 0.007]
    optimizers = [RMSprop, Adam, Adadelta]
    dropout = np.linspace(0.1, 0.5, 5).tolist()
    return {'batch_size': batches,
            'activation': activation,
            'lr':lr,
            'optimizer': optimizers,
            'drop': dropout}

# KerasClassifier 모델 구성하기
model = KerasClassifier(build_fn = build_model, verbose = 1)

# hyperparameters 변수 정의
hyperparameters = create_hyperparameter()

search = RandomizedSearchCV(estimator = model,
                            param_distributions = hyperparameters, cv = 3)

# 모델 훈련
search.fit(x_train, y_train)
score = search.score(x_test, y_test)
print(search.best_params_)              # {'optimizer': 'adadelta', 'drop': 0.2, 'batch_size': 20}
print("score : ", score)                # 0.9661999940872192

'''
