import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#====================Data and Splitting==============================

dados = pd.read_csv('database.csv')
dados = shuffle(dados)

x = dados.iloc[:, [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
y = dados.iloc[:, 4]

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.5, stratify=y)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp)

#====================Start of MLP Classifier================

# Definindo os parâmetros que queremos testar no GridSearch
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'max_iter': [1000, 1500, 2000],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

# Criando o modelo MLP
mlp_model = MLPClassifier(random_state=42)

# Configurando o GridSearchCV
grid_search = GridSearchCV(mlp_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Ajustando o GridSearch nos dados de treinamento
grid_search.fit(x_train, y_train)

# Obtendo os melhores parâmetros
best_params = grid_search.best_params_
print("Melhores Parâmetros: ", best_params)

# Treinando o modelo MLP com os melhores parâmetros encontrados
mlp_best = MLPClassifier(
    hidden_layer_sizes=best_params['hidden_layer_sizes'],
    activation=best_params['activation'],
    max_iter=best_params['max_iter'],
    learning_rate=best_params['learning_rate'],
    random_state=42
)
mlp_best.fit(x_train, y_train)

# Avaliando no conjunto de validação
opiniao_val = mlp_best.predict(x_val)
accuracy_val = accuracy_score(y_val, opiniao_val)
taxa_de_erro_val = 1 - accuracy_val
print("Acurácia no conjunto de validação: ", accuracy_val)
print("Taxa de erro no conjunto de validação: ", taxa_de_erro_val)

# Avaliando no conjunto de teste
opiniao_test = mlp_best.predict(x_test)
accuracy_test = accuracy_score(y_test, opiniao_test)
taxa_de_erro_test = 1 - accuracy_test
print("Acurácia no conjunto de teste: ", accuracy_test)
print("Taxa de erro no conjunto de teste: ", taxa_de_erro_test)

# Resumo dos Resultados do GridSearch
resultados = grid_search.cv_results_

# Taxa de erro média para cada combinação de hiperparâmetros
# taxa_de_erro = 1 - np.array(resultados['mean_test_score'])
# print("\nResumo dos Resultados do GridSearch (taxa de erro):")
# for i in range(len(taxa_de_erro)):
#     print(f"Combinação {i+1}: {resultados['params'][i]}, Taxa de erro: {taxa_de_erro[i]}")
