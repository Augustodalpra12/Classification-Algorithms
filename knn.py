# Importações iniciais

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier # knn
from sklearn.metrics import accuracy_score

#====================data and splitting==============================

dados = pd.read_csv('database.csv')
dados = shuffle(dados)

x = dados.iloc[:, [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
# y terá a coluna all_nba
y = dados.iloc[:, 4]

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.5, stratify=y)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp)

#===================== Prints=========================================

print("Treino")
#print(x_train)
# x_train.info()
# y_train.info()

#print("\nValidação")
#x_val.info()
#y_val.info()

#print("\nTeste")
#x_test.info()
#y_test.info()

#====================Start of KNN======================================
# Definindo os parâmetros que queremos testar no GridSearch
param_grid_knn = {
    'n_neighbors': list(range(1, 50)),
    'weights': ['uniform', 'distance']
}

# Criando o modelo KNN
KNN = KNeighborsClassifier()

# Configurando o GridSearchCV
grid_search = GridSearchCV(KNN, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Ajustando o GridSearch nos dados de treinamento e validação
grid_search.fit(x_train, y_train)

# Obtendo os melhores parâmetros
best_params = grid_search.best_params_
print("Melhores Parâmetros: ", best_params)

# Treinando o KNN com os melhores parâmetros encontrados
KNN_best = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
KNN_best.fit(x_train, y_train)

# Avaliando no conjunto de validação
opiniao = KNN_best.predict(x_val)
accuracy_val = accuracy_score(y_val, opiniao)
print("Acurácia no conjunto de validação: ", accuracy_val)

# Avaliando no conjunto de teste
opiniao_test = KNN_best.predict(x_test)
accuracy_test = accuracy_score(y_test, opiniao_test)
print("Acurácia no conjunto de teste: ", accuracy_test)

# Visualizando o erro para diferentes valores de K
resultados = grid_search.cv_results_
taxa_de_erro = [1 - score for score in resultados['mean_test_score']]

plt.figure(figsize=(11, 7))
plt.plot(range(1, 50), taxa_de_erro[:49], color='blue', linestyle='dashed', marker='o')
plt.xlabel('K')
plt.ylabel('Erro')
plt.show()