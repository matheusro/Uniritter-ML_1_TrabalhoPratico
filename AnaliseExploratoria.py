import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 100)

dataset = pd.read_csv('C:/dev/Uniritter_DS/MachineLearning_I/Uniritter-ML_1_TrabalhoPratico/data/dataset.csv', delimiter=',',  low_memory=False)

print("Amostra dos dados:")
print(dataset.head())
shape = dataset.shape
print("Total de Linhas e Colunas: ", shape)
sumario = dataset.describe()
print("Informações estatisticas do Dataset: ")
print(sumario)
print('\n')

print("Histograma com a distibuicao da classe Cover Type")
dataset.Cover_Type.hist()
plt.title('Histograma com a distibuicao da classe Cover Type')
plt.xlabel('Classe: Cover Type')
plt.ylabel('Frequencia')

