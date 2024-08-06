import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#====================Data and Splitting==============================

dados = pd.read_csv('database.csv')
dados = shuffle(dados)

x = dados.iloc[:, [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
y = dados.iloc[:, 4]

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.5, stratify=y)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp)

#====================Start of Naive Bayes Classifier================

# Criando e treinando o modelo de Naive Bayes
naive_bayes = GaussianNB()
naive_bayes.fit(x_train, y_train)

# Avaliando no conjunto de validação
opiniao_val = naive_bayes.predict(x_val)
accuracy_val = accuracy_score(y_val, opiniao_val)
taxa_de_erro_val = 1 - accuracy_val
print("Acurácia no conjunto de validação: ", accuracy_val)
print("Taxa de erro no conjunto de validação: ", taxa_de_erro_val)

# Avaliando no conjunto de teste
opiniao_test = naive_bayes.predict(x_test)
accuracy_test = accuracy_score(y_test, opiniao_test)
taxa_de_erro_test = 1 - accuracy_test
print("Acurácia no conjunto de teste: ", accuracy_test)
print("Taxa de erro no conjunto de teste: ", taxa_de_erro_test)
