import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#====================Data and Splitting==============================

dados = pd.read_csv('database.csv')
dados = shuffle(dados)

x = dados.iloc[:, [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
y = dados.iloc[:, 4]

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.5, stratify=y)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp)

#====================Start of SVM Classifier================

# Definindo os parâmetros que queremos testar no GridSearch
param_grid = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.1, 1, 10, 100, 1000]
}

# Criando o modelo SVM
svm_model = SVC(random_state=42)

# Configurando o GridSearchCV
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Ajustando o GridSearch nos dados de treinamento
grid_search.fit(x_train, y_train)

# Obtendo os melhores parâmetros
best_params = grid_search.best_params_
print("Melhores Parâmetros: ", best_params)

# Treinando o modelo SVM com os melhores parâmetros encontrados
svm_best = SVC(kernel=best_params['kernel'], C=best_params['C'], random_state=42)
svm_best.fit(x_train, y_train)

# Avaliando no conjunto de validação
opiniao_val = svm_best.predict(x_val)
accuracy_val = accuracy_score(y_val, opiniao_val)
print("Acurácia no conjunto de validação: ", accuracy_val)

# Avaliando no conjunto de teste
opiniao_test = svm_best.predict(x_test)
accuracy_test = accuracy_score(y_test, opiniao_test)
print("Acurácia no conjunto de teste: ", accuracy_test)

# Visualizando os resultados do GridSearch
resultados = grid_search.cv_results_

# Convertendo os resultados de erro (1 - acurácia) para cada combinação de hiperparâmetros
taxa_de_erro = 1 - np.array(resultados['mean_test_score'])

# Convertendo as colunas dos parâmetros para arrays de numpy
cs = np.array(resultados['param_C'].data, dtype=float)
kernels = np.array(resultados['param_kernel'].data)

# Criando um gráfico para cada valor de kernel
plt.figure(figsize=(11, 7))

# Iterando sobre os valores de kernel
for kernel in np.unique(kernels):
    # Filtrando os resultados para o valor atual de kernel
    mask = (kernels == kernel)
    taxa_de_erro_filtrada = taxa_de_erro[mask]
    cs_filtrados = cs[mask]

    # Ordenando os valores por C
    sorted_indices = np.argsort(cs_filtrados)
    cs_filtrados = cs_filtrados[sorted_indices]
    taxa_de_erro_filtrada = taxa_de_erro_filtrada[sorted_indices]

    # Plotando a taxa de erro para cada valor de C
    plt.plot(
        cs_filtrados,
        taxa_de_erro_filtrada,
        marker='o',
        label=f'Kernel: {kernel}'
    )

plt.xlabel('C (Regularização)')
plt.ylabel('Erro')
plt.xscale('log')
plt.title('Erro vs C para diferentes valores de Kernel')
plt.legend()
plt.show()
