import matplotlib.pyplot as plt
import numpy as np
from slugify import slugify

def bars(title, categories, values):
  plt.bar(categories, values)
 
  plt.xlabel('Descrição')
  plt.ylabel('Valores')
  plt.title(title)
  
  plt.savefig(slugify(title))

def confusionMatrixCompare(title, rnaMatrix, svmMatrix):
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

  # RNA
  axes[0].imshow(rnaMatrix, interpolation='nearest', cmap=plt.cm.Blues)
  axes[0].set_title('RNA Matriz de Confusão')
  plt.colorbar(ax=axes[0], mappable=axes[0].images[0], orientation='vertical')
  axes[0].set_xticks(np.arange(2))
  axes[0].set_yticks(np.arange(2))
  axes[0].set_xticklabels(['Negativo', 'Positivo'])
  axes[0].set_yticklabels(['Negativo', 'Positivo'])
  axes[0].set_xlabel('Rótulos Previstos')
  axes[0].set_ylabel('Rótulos Verdadeiros')

  # SVM
  axes[1].imshow(svmMatrix, interpolation='nearest', cmap=plt.cm.Blues)
  axes[1].set_title('SVM Matriz de Confusão')
  plt.colorbar(ax=axes[1], mappable=axes[1].images[0], orientation='vertical')
  axes[1].set_xticks(np.arange(2))
  axes[1].set_yticks(np.arange(2))
  axes[1].set_xticklabels(['Negativo', 'Positivo'])
  axes[1].set_yticklabels(['Negativo', 'Positivo'])
  axes[1].set_xlabel('Rótulos Previstos')
  axes[1].set_ylabel('Rótulos Verdadeiros')

  plt.suptitle(title)
  plt.tight_layout()
  plt.savefig(slugify(title))