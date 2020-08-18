'''
중/저이미지(화질) * 1000명(ID) * 악세사리(비착용 : S001, 일반안경 : S002, 뿔테안경 : S003) 

* 조명 (L1, L2, L3, L4, L8, L9, L19, L20, L22, L23, L25, L26) * 표정 (무표정 : E01, 웃음 : E02)

* 각도 (C6, C7, C8) = 432,000장

'''



# 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten


model = Sequential()

model.add(Conv2D(10, (2, 2), input_shape = (2, 2, 1 ), activation = 'relu', padding = 'same'))
model.add(Conv2D(50, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Conv2D(50, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Conv2D(50, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Conv2D(50,(2, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Conv2D(50, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Conv2D(50,(2, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Conv2D(50, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Conv2D(50, (2, 2),activation = 'relu', padding = 'same'))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(3, activation = 'softmax'))

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


#3. compile, fit

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

hist = model.fit(x_train, y_train, epochs =100, batch_size= 64,
                validation_split = 0.2, verbose = 2,
                callbacks = [es])



# evaluate

loss, acc = model.evaluate(x_test, y_test, batch_size = 64)

print('loss: ', loss )
print('acc: ', acc)

# 그래프

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

plt.show()