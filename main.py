import rna
import ridge
import svm
from keras import losses
import chart
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preProcessingData(df):
  df = df[df['Volume'] != 0]
  df = df.dropna()
  df = df.reset_index()
  df['Target'] = df[['Close']].shift(-1)
  df = df.iloc[:-1]

  sc = MinMaxScaler(feature_range=(0,1))
  scaled_df = pd.DataFrame(sc.fit_transform(df.drop(columns=['Date', 'Adj Close'])), columns=df.drop(columns=['Date', 'Adj Close']).columns)

  return df, scaled_df

def plotPrevistoXRealizado(model, x, scaled_df, df, title):
  predict_price = model.predict(x)
  Predict = []
  for i in predict_price:
    Predict.append(i[0])

  close = scaled_df['Close']
  
  df_predict = df[['Date']]
  df_predict['Real'] = close
  df_predict['Previsto'] = Predict
  chart.line(title, df_predict)

def main():
  ticks = [
    {
      'name': 'VALE3',
      'file': 'data/VALE3.SA.csv'
    },
    {
      'name': 'MGLU3',
      'file': 'data/MGLU3.SA.csv'
    },
    {
      'name': 'CSAN3.SA.csv',
      'file': 'data/CSAN3.SA.csv'
    },
    {
      'name': 'APPLE',
      'file': 'data/AAPL.csv'
    }
  ]
  for tick in ticks:
    df = pd.read_csv(tick['file'])

    dfTrain = df[df['Date'] < '2023-01-01']
    dfTrain, scaled_dfTrain = preProcessingData(dfTrain)

    xTrain = scaled_dfTrain[['Open', 'High', 'Low', 'Close', 'Volume']]
    yTrain = scaled_dfTrain[['Target']]

    dfTest = df[df['Date'] >= '2023-01-01']
    dfTest, scaled_dfTest = preProcessingData(dfTest)

    xTest = scaled_dfTest[['Open', 'High', 'Low', 'Close', 'Volume']]
    yTest = scaled_dfTest[['Target']]
  

    rnaModel = rna.Model(
      optimizer='Adam',
      loss=losses.mse,
      epochs=1000,
      batch_size=64,
      xTrain=xTrain,
      yTrain=yTrain,
      xTest=xTest,
      yTest=yTest
    )
    rnaStatistics = rnaModel.process()
    plotPrevistoXRealizado(rnaModel.model, xTest, scaled_dfTest, dfTest, 'Previsto x Realizado: RNA ' + tick['name'])

    ridgeModel = ridge.Model(
      alpha=1.0,
      xTrain=xTrain,
      yTrain=yTrain,
      xTest=xTest,
      yTest=yTest
    )
    ridgeStatistics = ridgeModel.process()
    plotPrevistoXRealizado(ridgeModel.model, xTest, scaled_dfTest, dfTest, 'Previsto x Realizado: RIDGE ' + tick['name'])

    svmModel = svm.Model(
      penalty=1.0,
      xTrain=xTrain,
      yTrain=yTrain,
      xTest=xTest,
      yTest=yTest
    )
    svmStatistics = svmModel.process()
    plotPrevistoXRealizado(ridgeModel.model, xTest, scaled_dfTest, dfTest, 'Previsto x Realizado: SVM ' + tick['name'])

    chart.bars(
      title='Erro médio absoluto comparação: ' + tick['name'],
      categories=['RNA', 'RR', 'SVM'],
      values=[
        rnaStatistics['mean_absolute_error'],
        ridgeStatistics['mean_absolute_error'],
        svmStatistics['mean_absolute_error'],
      ]
    )

    chart.bars(
      title='Erro Quadrático Médio comparação: ' + tick['name'],
      categories=['RNA', 'RR', 'SVM'],
      values=[
        rnaStatistics['mean_squared_error'],
        ridgeStatistics['mean_squared_error'],
        svmStatistics['mean_squared_error'],
      ]
    )

    chart.bars(
      title='Coeficiente de determinação: ' + tick['name'],
      categories=['RNA', 'RR', 'SVM'],
      values=[
        rnaStatistics['r2_score'],
        ridgeStatistics['r2_score'],
        svmStatistics['r2_score'],
      ]
    )

main()