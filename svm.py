from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from sklearn.svm import SVR


class Model:
  def __init__(self, penalty, xTrain, yTrain, xTest, yTest):
    self.penalty = penalty
    self.xTrain = xTrain
    self.yTrain = yTrain
    self.xTest = xTest
    self.yTest = yTest
  
  def process(self):
    model = SVR(C=self.penalty)
    model.fit(self.xTrain, self.yTrain)
    self.model = model

    predict = model.predict(self.xTest)

    return {
      'mean_absolute_error': mean_absolute_error(self.yTest, predict),
      'mean_squared_error': mean_squared_error(self.yTest, predict),
      'r2_score': r2_score(self.yTest, predict)
    }