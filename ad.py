import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#====================Data and Splitting==============================

dados = pd.read_csv('database.csv')
dados = shuffle(dados)

x = dados.iloc[:, [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
y = dados.iloc[:, 4]

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.5, stratify=y)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp)

#====================Start of Decision Tree Classifier================

# Definindo os parâmetros que queremos testar no GridSearch
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 10]
}

# Criando o modelo de Árvore de Decisão
decision_tree = DecisionTreeClassifier(random_state=42)

# Configurando o GridSearchCV
grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Ajustando o GridSearch nos dados de treinamento
grid_search.fit(x_train, y_train)

# Obtendo os melhores parâmetros
best_params = grid_search.best_params_
print("Melhores Parâmetros: ", best_params)

# Treinando o modelo de Árvore de Decisão com os melhores parâmetros encontrados
decision_tree_best = DecisionTreeClassifier(
    criterion=best_params['criterion'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)
decision_tree_best.fit(x_train, y_train)

# Avaliando no conjunto de validação
opiniao_val = decision_tree_best.predict(x_val)
accuracy_val = accuracy_score(y_val, opiniao_val)
print("Acurácia no conjunto de validação: ", accuracy_val)

# Avaliando no conjunto de teste
opiniao_test = decision_tree_best.predict(x_test)
accuracy_test = accuracy_score(y_test, opiniao_test)
print("Acurácia no conjunto de teste: ", accuracy_test)

# Visualizando os resultados do GridSearch
resultados = grid_search.cv_results_

# Convertendo os resultados de erro (1 - acurácia) para cada combinação de hiperparâmetros
taxa_de_erro = 1 - np.array(resultados['mean_test_score'])

# Convertendo as colunas dos parâmetros para arrays de numpy
max_depths = np.array(resultados['param_max_depth'].data, dtype=float)
min_samples_splits = np.array(resultados['param_min_samples_split'].data, dtype=int)

# Criando um gráfico para cada valor de max_depth
plt.figure(figsize=(11, 7))

# Iterando sobre os valores de max_depth
for depth in np.unique(max_depths):
    # Filtrando os resultados para o valor atual de max_depth
    mask = (max_depths == depth)
    taxa_de_erro_filtrada = taxa_de_erro[mask]
    min_samples_split_filtrados = min_samples_splits[mask]

    # Ordenando os valores por min_samples_split
    sorted_indices = np.argsort(min_samples_split_filtrados)
    min_samples_split_filtrados = min_samples_split_filtrados[sorted_indices]
    taxa_de_erro_filtrada = taxa_de_erro_filtrada[sorted_indices]

    # Plotando a taxa de erro para cada valor de min_samples_split
    plt.plot(
        min_samples_split_filtrados,
        taxa_de_erro_filtrada,
        marker='o',
        label=f'Depth: {depth}'
    )

plt.xlabel('Min Samples Split')
plt.ylabel('Erro')
plt.title('Erro vs Min Samples Split para diferentes valores de Max Depth')
plt.legend()
plt.show()