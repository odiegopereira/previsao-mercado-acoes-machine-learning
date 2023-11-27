from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import Ridge


class Model:
  def __init__(self, title, x, y):
    self.title = title
    self.x = x
    self.y = y
  
  def process(self):
    X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
  

    model = Ridge()
    model.fit(X_train, y_train)
    self.model = model

    accuracy = model.score(X_test, y_test)

    return {
      'accuracy': accuracy,
    }