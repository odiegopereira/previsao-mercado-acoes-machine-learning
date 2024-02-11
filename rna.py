import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 


class Model:
  def __init__(self, optimizer, loss, epochs, batch_size, xTrain, yTrain, xTest, yTest):
    self.optimizer = optimizer
    self.loss = loss
    self.epochs = epochs
    self.batch_size = batch_size
    self.xTrain = xTrain
    self.yTrain = yTrain
    self.xTest = xTest
    self.yTest = yTest
  
  def process(self):
  
    # Create the model
    model = Sequential()
    model.add(Dense(units=12, input_dim=self.xTrain.shape[1], activation='relu'))
    model.add(Dense(units=12, activation='relu'))
    model.add(Dense(units=3, activation='relu'))
    model.add(Dense(units=1, activation='relu'))

    model.compile(optimizer=self.optimizer, loss=self.loss)
    model.fit(self.xTrain, self.yTrain, epochs=self.epochs, batch_size=self.batch_size)
    
    self.model = model

    predict = model.predict(self.xTest)
    
    return {
      'mean_absolute_error': mean_absolute_error(self.yTest, predict),
      'mean_squared_error': mean_squared_error(self.yTest, predict),
      'r2_score': r2_score(self.yTest, predict)
    }