import rna
import ridge
from keras import losses
import chart
import pandas as pd
import talib as ta
from talib import MA_Type
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def getVariationClass(value):
  if value <= 0: return 0
  else: return 1

def prepareDataframe(df):
  df = df[df['Volume'] != 0]
  df = df.dropna()
  df = df.reset_index()
  df['Target'] = df[['Close']].shift(-1)
  df = df.iloc[:-1]

  return df

def appendIndicators(df):
  # Média móvel exponencial do Close, Open, High e Low
  df['EMAC'] = ta.EMA(df['Close'], timeperiod=3)
  df['EMAO'] = ta.EMA(df['Open'], timeperiod=3)
  df['EMAH'] = ta.EMA(df['High'], timeperiod=3)
  df['EMAL'] = ta.EMA(df['Low'], timeperiod=3)
  
  df['RSI'] = ta.RSI(df['EMAC'], timeperiod=14) # Relative Strength Index
  df['WILLR'] = ta.WILLR(df['EMAH'], df['EMAL'], df['EMAC'], timeperiod=14) # Williams %R
  df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = ta.MACD(df['EMAC'], fastperiod=14, slowperiod=24, signalperiod=14) #Moving Average Convergence Divergence
  df['OBV'] = ta.OBV(df['EMAC'], df['Volume']) # On Balance Volume
  df['ROC'] = ta.ROC(df['EMAC'], timeperiod=14) #Price Rate of Change
  df['FASTK'], df['FASTD'] = ta.STOCHRSI(df['EMAC'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0) #Stochastic Oscillator
  
  df.fillna(method="ffill", inplace= True)
  df.fillna(method="bfill",inplace= True)
  df = df.reset_index()
  
  return df

def interactive_plot(df, title):
  plt.figure(figsize=(10, 6))
  plt.title(title)
  
  for i in df.columns[1:]:
      plt.plot(df['Date'], df[i], label=i)

  plt.xlabel('Date')
  plt.ylabel('Values')
  plt.legend()
  plt.grid(True)
  plt.savefig('hello_world')


def main():
  df = pd.read_csv('MGLU3.SA.csv')
  df = prepareDataframe(df)
  #df = appendIndicators(df)

  sc = MinMaxScaler(feature_range=(0,1))
  scaled_df = pd.DataFrame(sc.fit_transform(df.drop(columns=['Date', 'Adj Close'])), columns=df.drop(columns=['Date', 'Adj Close']).columns)

  x = scaled_df[['Open', 'High', 'Low', 'Close', 'Volume']]
  y = scaled_df[['Target']]
 

  # rnaModel = rna.Model(
  #   title='Sem indicadores',
  #   optimizer='Adam',
  #   loss=losses.categorical_hinge,
  #   epochs=500,
  #   batch_size=64,
  #   x=x,
  #   y=y
  # )
  # rnaStatistics = rnaModel.process()
  # print(rnaStatistics)

  ridgeModel = ridge.Model(
    title='Sem indicadores',
    x=x,
    y=y
  )
  ridgeStatistics = ridgeModel.process()

  predict_price = ridgeModel.model.predict(x)
  Predict = []
  for i in predict_price:
    Predict.append(i[0])

  close = scaled_df['Close']
  
  df_predict = df[['Date']]
  df_predict['Close'] = close
  df_predict['Prediction'] = Predict
  interactive_plot(df_predict, 'Teste')

  # chart.bars(
  #   title='Acurácia: Comparação com Indicadores',
  #   categories=['RNA', 'RNA Chute', 'SVM', 'SVM Chute'],
  #   values=[
  #     rnaWithoutIndicatorsStatistics['accuracy']*100,
  #     rnaWithoutIndicatorsStatistics['accuracy_most_frequently']*100,
  #     svmWithoutIndicatorsStatistics['accuracy']*100,
  #     svmWithoutIndicatorsStatistics['accuracy_most_frequently']*100
  #   ]
  # )
  



main()