from keras import layers, models
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
 
class AE(models.Model):
  def __init__(self, x_nodes=784, z_dim=36):
    x = layers.Input(shape=(x_nodes,))
    z = layers.Dense(z_dim, activation='relu')(x)
    y = layers.Dense(x_nodes, activation='sigmoid')(z)
    
    super().__init__(x, y)
    
    self.x = x
    self.z = z
    self.z_dim = z_dim
    
    self.compile(optimizer='adadelta', loss='binary_crossentropy',
                metrics=['accuracy'])
    
  def get_encoder(self):
    return models.Model(self.x, self.z)
  
  def get_decoder(self):
    z = layers.Input(shape=(self.z_dim,))
    y = self.layers[-1](z)
    return models.Model(z, y)
  
 
(x_train, _), (x_test, _) = mnist.load_data()
 
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

print(x_train.shape)
print(x_test.shape)
 
def show_ae(ae, sample_size=10):
  encoder = ae.get_encoder()
  decoder = ae.get_decoder()
  
  encoded = encoder.predict(x_test, batch_size=sample_size)
  decoded = decoder.predict(encoded)
  
  plt.figure(figsize=(20, 6))
  
  for i in range(sample_size):
    ax = plt.subplot(3, sample_size, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(3, sample_size, i+1+sample_size)
    plt.stem(encoded[i].reshape(-1))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(3, sample_size, i+1+sample_size*2)
    plt.imshow(decoded[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
  plt.show()
 
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)
 
def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)
  
def main():
  x_nodes = 784
  z_dim = 36
  
  autoencoder = AE(x_nodes, z_dim)
  
  history = autoencoder.fit(x_train, x_train, epochs=100, batch_size=256,
                           shuffle=True, validation_data=(x_test, x_test),
                           verbose=2)
  
  plot_acc(history)
  plt.show()
  plot_loss(history)
  plt.show()
  
  show_ae(autoencoder)
  
if __name__ == '__main__':
  main()

