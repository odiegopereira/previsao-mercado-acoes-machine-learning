import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt


class Model:
  def __init__(self, title, optimizer, loss, epochs, batch_size, x, y):
    self.title = title
    self.optimizer = optimizer
    self.loss = loss
    self.epochs = epochs
    self.batch_size = batch_size
    self.x = x
    self.y = y
  
  def process(self):
    X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
  

    model = Sequential()
    model.add(Dense(units=12, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(units=12, activation='relu'))
    model.add(Dense(units=3, activation='relu'))
    model.add(Dense(units=1, activation='relu'))

    model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
    self.model = model

    y_pred = model.predict(X_test)
    #accuracy = model.score(y_test, y_pred)

    return {
      'accuracy': 0,
    }