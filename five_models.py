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
from sklearn.naive_bayes import BernoulliNB # nb
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
    param_grid_knn = {
        'n_neighbors': list(range(1, 50)),
        'weights': ['uniform', 'distance']
    }

    best_accuracy = 0
    best_params = {}

    for n_neighbors in param_grid_knn['n_neighbors']:
        for weight in param_grid_knn['weights']:
            # Treinando o KNN com os parâmetros atuais
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight)
            knn.fit(x_train, y_train)
            
            # Avaliando no conjunto de validação
            opiniao = knn.predict(x_val)
            accuracy_val = accuracy_score(y_val, opiniao)
            
            # Se a acurácia atual for melhor que a melhor acurácia, atualize os melhores parâmetros
            if accuracy_val > best_accuracy:
                best_accuracy = accuracy_val
                best_params = {'n_neighbors': n_neighbors, 'weights': weight}

    # Exibindo os melhores parâmetros
    print("Melhores Parâmetros: ", best_params)
    print("Melhor Acurácia no conjunto de validação: ", best_accuracy)

    # Treinando o modelo final com os melhores parâmetros
    KNN_best = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
    KNN_best.fit(x_train, y_train)

    # Avaliando no conjunto de teste
    opiniao_test = KNN_best.predict(x_test)
    accuracy_test = accuracy_score(y_test, opiniao_test)
    print("Acurácia no conjunto de teste: ", accuracy_test)

    #====================Start of SVM Classifier================
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'C': [0.1, 1, 10, 100, 1000]
    }

    best_accuracy = 0
    best_params = {}

    for kernel in param_grid['kernel']:
        for C in param_grid['C']:
            # Treinando o SVM com os parâmetros atuais
            svm_model = SVC(kernel=kernel, C=C, random_state=42)
            svm_model.fit(x_train, y_train)
            
            # Avaliando no conjunto de validação
            opiniao = svm_model.predict(x_val)
            accuracy_val = accuracy_score(y_val, opiniao)
            
            # Se a acurácia atual for melhor que a melhor acurácia, atualize os melhores parâmetros
            if accuracy_val > best_accuracy:
                best_accuracy = accuracy_val
                best_params = {'kernel': kernel, 'C': C}

    # Exibindo os melhores parâmetros
    print("Melhores Parâmetros: ", best_params)
    print("Melhor Acurácia no conjunto de validação: ", best_accuracy)

    # Treinando o modelo final com os melhores parâmetros
    svm_best = SVC(kernel=best_params['kernel'], C=best_params['C'], random_state=42)
    svm_best.fit(x_train, y_train)

    # Avaliando no conjunto de teste
    opiniao_test = svm_best.predict(x_test)
    accuracy_test = accuracy_score(y_test, opiniao_test)
    print("Acurácia no conjunto de teste: ", accuracy_test)

    #====================Start of Decision Tree Classifier================
    # Definindo os parâmetros que queremos testar
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 10]
    }

    best_accuracy = 0
    best_params = {}

    # Laço para testar todas as combinações de parâmetros
    for criterion in param_grid['criterion']:
        for max_depth in param_grid['max_depth']:
            for min_samples_split in param_grid['min_samples_split']:
                for min_samples_leaf in param_grid['min_samples_leaf']:
                    # Treinando o modelo de Árvore de Decisão com os parâmetros atuais
                    decision_tree = DecisionTreeClassifier(
                        criterion=criterion,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=42
                    )
                    decision_tree.fit(x_train, y_train)
                    
                    # Avaliando no conjunto de validação
                    opiniao_val = decision_tree.predict(x_val)
                    accuracy_val = accuracy_score(y_val, opiniao_val)
                    
                    # Se a acurácia atual for melhor que a melhor acurácia, atualize os melhores parâmetros
                    if accuracy_val > best_accuracy:
                        best_accuracy = accuracy_val
                        best_params = {
                            'criterion': criterion,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf
                        }

    # Exibindo os melhores parâmetros
    print("Melhores Parâmetros: ", best_params)
    print("Melhor Acurácia no conjunto de validação: ", best_accuracy)

    # Treinando o modelo final com os melhores parâmetros
    decision_tree_best = DecisionTreeClassifier(
        criterion=best_params['criterion'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42
    )
    decision_tree_best.fit(x_train, y_train)

    # Avaliando no conjunto de teste
    opiniao_test = decision_tree_best.predict(x_test)
    accuracy_test = accuracy_score(y_test, opiniao_test)
    print("Acurácia no conjunto de teste: ", accuracy_test)

    #====================Start of MLP Classifier================
    param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'max_iter': [1000, 1500, 2000],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
    }

    best_accuracy = 0
    best_params = {}

    # Laço para testar todas as combinações de parâmetros
    for hidden_layer_sizes in param_grid['hidden_layer_sizes']:
        for activation in param_grid['activation']:
            for max_iter in param_grid['max_iter']:
                for learning_rate in param_grid['learning_rate']:
                    # Treinando o modelo MLP com os parâmetros atuais
                    mlp_model = MLPClassifier(
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation,
                        max_iter=max_iter,
                        learning_rate=learning_rate,
                        random_state=42
                    )
                    mlp_model.fit(x_train, y_train)
                    
                    # Avaliando no conjunto de validação
                    opiniao_val = mlp_model.predict(x_val)
                    accuracy_val = accuracy_score(y_val, opiniao_val)
                    
                    # Se a acurácia atual for melhor que a melhor acurácia, atualize os melhores parâmetros
                    if accuracy_val > best_accuracy:
                        best_accuracy = accuracy_val
                        best_params = {
                            'hidden_layer_sizes': hidden_layer_sizes,
                            'activation': activation,
                            'max_iter': max_iter,
                            'learning_rate': learning_rate
                        }

    # Exibindo os melhores parâmetros
    print("Melhores Parâmetros: ", best_params)
    print("Melhor Acurácia no conjunto de validação: ", best_accuracy)

    # Treinando o modelo final com os melhores parâmetros
    mlp_best = MLPClassifier(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        activation=best_params['activation'],
        max_iter=best_params['max_iter'],
        learning_rate=best_params['learning_rate'],
        random_state=42
    )
    mlp_best.fit(x_train, y_train)

    # Avaliando no conjunto de teste
    opiniao_test = mlp_best.predict(x_test)
    accuracy_test = accuracy_score(y_test, opiniao_test)
    taxa_de_erro_test = 1 - accuracy_test
    print("Acurácia no conjunto de teste: ", accuracy_test)
    print("Taxa de erro no conjunto de teste: ", taxa_de_erro_test)

    #====================Start of Naive Bayes Classifier================
    print("\nNB", file=f)

    # Criando e treinando o modelo de Naive Bayes
    naive_bayes = BernoulliNB()
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
