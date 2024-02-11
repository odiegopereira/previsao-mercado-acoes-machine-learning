import matplotlib.pyplot as plt
import numpy as np
from slugify import slugify

def bars(title, categories, values):
  plt.clf()
  plt.bar(categories, values)
 
  plt.xlabel('Descrição')
  plt.ylabel('Valores')
  plt.title(title)

  for i, value in enumerate(values):
    plt.text(categories[i], value, str(round(value, 5)), ha='center', va='bottom')
  
  plt.savefig(slugify(title))

def line(title, df):
  plt.clf()
  plt.figure(figsize=(10, 6))
  plt.title(title)
  
  for i in df.columns[1:]:
    plt.plot(df.index, df[i], label=i)

  plt.xlabel('Índice')
  plt.ylabel('Valores')
  plt.legend()
  plt.grid(True)
  plt.savefig(slugify(title))