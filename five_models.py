# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier # knn
from sklearn.svm import SVC # svc
from sklearn.tree import DecisionTreeClassifier # ad
from sklearn.neural_network import MLPClassifier # mlp
from sklearn.naive_bayes import GaussianNB # nb
from sklearn.metrics import accuracy_score

# Abrindo o arquivo para escrita
with open('results.txt', 'w') as f:

    #====================data and splitting==============================
    dados = pd.read_csv('database.csv')
    dados = shuffle(dados)

    x = dados.iloc[:, [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    # y terá a coluna all_nba
    y = dados.iloc[:, 4]

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.5, stratify=y)
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp)

    #====================Start of KNN======================================
    print("KNN", file=f)

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
    grid_search.fit(x_val, y_val)

    # Obtendo os melhores parâmetros
    best_params = grid_search.best_params_
    print("Melhores Parâmetros: ", best_params, file=f)

    # Treinando o KNN com os melhores parâmetros encontrados
    KNN_best = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
    KNN_best.fit(x_train, y_train)

    # Avaliando no conjunto de validação
    opiniao = KNN_best.predict(x_val)
    accuracy_val = accuracy_score(y_val, opiniao)
    print("Acurácia no conjunto de validação: ", accuracy_val, file=f)

    # Avaliando no conjunto de teste
    opiniao_test = KNN_best.predict(x_test)
    accuracy_test = accuracy_score(y_test, opiniao_test)
    print("Acurácia no conjunto de teste: ", accuracy_test, file=f)

    #====================Start of SVM Classifier================
    print("\nSVM", file=f)

    # Definindo os parâmetros que queremos testar no GridSearch
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'C': [0.1, 1, 10, 100, 1000]
    }

    # Criando o modelo SVM
    svm_model = SVC(random_state=42)

    # Configurando o GridSearchCV
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

    # Ajustando o GridSearch nos dados de validação
    grid_search.fit(x_val, y_val)

    # Obtendo os melhores parâmetros
    best_params = grid_search.best_params_
    print("Melhores Parâmetros: ", best_params, file=f)

    # Treinando o modelo SVM com os melhores parâmetros encontrados
    svm_best = SVC(kernel=best_params['kernel'], C=best_params['C'], random_state=42)
    svm_best.fit(x_train, y_train)

    # Avaliando no conjunto de validação
    opiniao_val = svm_best.predict(x_val)
    accuracy_val = accuracy_score(y_val, opiniao_val)
    print("Acurácia no conjunto de validação: ", accuracy_val, file=f)

    # Avaliando no conjunto de teste
    opiniao_test = svm_best.predict(x_test)
    accuracy_test = accuracy_score(y_test, opiniao_test)
    print("Acurácia no conjunto de teste: ", accuracy_test, file=f)

    #====================Start of Decision Tree Classifier================
    print("\nAD", file=f)

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

    # Ajustando o GridSearch nos dados de validação
    grid_search.fit(x_val, y_val)

    # Obtendo os melhores parâmetros
    best_params = grid_search.best_params_
    print("Melhores Parâmetros: ", best_params, file=f)

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
    print("Acurácia no conjunto de validação: ", accuracy_val, file=f)

    # Avaliando no conjunto de teste
    opiniao_test = decision_tree_best.predict(x_test)
    accuracy_test = accuracy_score(y_test, opiniao_test)
    print("Acurácia no conjunto de teste: ", accuracy_test, file=f)

    #====================Start of MLP Classifier================
    print("\nMLP", file=f)

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

    # Ajustando o GridSearch nos dados de validação
    grid_search.fit(x_val, y_val)

    # Obtendo os melhores parâmetros
    best_params = grid_search.best_params_
    print("Melhores Parâmetros: ", best_params, file=f)

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
    print("Acurácia no conjunto de validação: ", accuracy_val, file=f)
    print("Taxa de erro no conjunto de validação: ", taxa_de_erro_val, file=f)

    # Avaliando no conjunto de teste
    opiniao_test = mlp_best.predict(x_test)
    accuracy_test = accuracy_score(y_test, opiniao_test)
    taxa_de_erro_test = 1 - accuracy_test
    print("Acurácia no conjunto de teste: ", accuracy_test, file=f)
    print("Taxa de erro no conjunto de teste: ", taxa_de_erro_test, file=f)

    #====================Start of Naive Bayes Classifier================
    print("\nNB", file=f)

    # Criando e treinando o modelo de Naive Bayes
    naive_bayes = GaussianNB()
    naive_bayes.fit(x_train, y_train)

    # Avaliando no conjunto de validação
    opiniao_val = naive_bayes.predict(x_val)
    accuracy_val = accuracy_score(y_val, opiniao_val)
    taxa_de_erro_val = 1 - accuracy_val
    print("Acurácia no conjunto de validação: ", accuracy_val, file=f)
    print("Taxa de erro no conjunto de validação: ", taxa_de_erro_val, file=f)

    # Avaliando no conjunto de teste
    opiniao_test = naive_bayes.predict(x_test)
    accuracy_test = accuracy_score(y_test, opiniao_test)
    taxa_de_erro_test = 1 - accuracy_test
    print("Acurácia no conjunto de teste: ", accuracy_test, file=f)
    print("Taxa de erro no conjunto de teste: ", taxa_de_erro_test, file=f)
