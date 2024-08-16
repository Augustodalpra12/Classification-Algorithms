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
from sklearn.metrics import confusion_matrix
import scipy.stats as stats # kruskal-wallis
from scipy.stats import mannwhitneyu # Mann-Whitney 

# Abrindo o arquivo para escrita
with open('results.txt', 'w') as txt:
    accuracy_list = []
    accuracy_list_mult = []
    
    for i in range(1, 2):
        print('Iteração ', i, file=txt)
        
        #====================data and splitting==============================
        dados = pd.read_csv('database.csv')
        dados = shuffle(dados)

        x = dados.iloc[:, [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
        y = dados.iloc[:, 4]

        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.5, stratify=y)
        x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp)

        accuracy = []

        #====================Start of KNN======================================
        print("\nKNN", file=txt)

        param_grid_knn = {
            'n_neighbors': list(range(1, 50)),
            'weights': ['uniform', 'distance']
        }

        best_accuracy = 0
        best_params = {}

        for n_neighbors in param_grid_knn['n_neighbors']:
            for weight in param_grid_knn['weights']:
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight)
                knn.fit(x_train, y_train)
                
                opiniao = knn.predict(x_val)
                accuracy_val = accuracy_score(y_val, opiniao)
                
                if accuracy_val > best_accuracy:
                    best_accuracy = accuracy_val
                    best_params = {'n_neighbors': n_neighbors, 'weights': weight}

        print("Melhores Parâmetros: ", best_params, file=txt)
        print("Melhor Acurácia no conjunto de validação: ", best_accuracy, file=txt)

        KNN_best = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
        KNN_best.fit(x_train, y_train)
        # Para os multiclassificadores
        knn_probs = KNN_best.predict_proba(x_test)


        opiniao_test = KNN_best.predict(x_test)
        accuracy_test = accuracy_score(y_test, opiniao_test)
        print("Acurácia no conjunto de teste: ", accuracy_test, file=txt)
        print(accuracy_test) # vai pro csv na coluna KNN
        print("Matriz de Confusão\n ",confusion_matrix(y_test, opiniao_test), file=txt)
        TN, FP, FN, TP = confusion_matrix(y_test, opiniao_test).ravel()
        print("Sensibilidade: ",(TP/(TP+FN)), file=txt)
        print("Especificade: ",(TN/(FP+TN)), file=txt)

        accuracy.append(accuracy_test)

        #====================Start of SVM Classifier================
        print("\nSVM", file=txt)

        param_grid = {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'C': [0.1, 1, 10, 100, 1000]
        }

        best_accuracy = 0
        best_params = {}

        for kernel in param_grid['kernel']:
            for C in param_grid['C']:
                svm_model = SVC(kernel=kernel, C=C, random_state=42)
                svm_model.fit(x_train, y_train)
                
                opiniao = svm_model.predict(x_val)
                accuracy_val = accuracy_score(y_val, opiniao)
                
                if accuracy_val > best_accuracy:
                    best_accuracy = accuracy_val
                    best_params = {'kernel': kernel, 'C': C}

        print("Melhores Parâmetros: ", best_params, file=txt)
        print("Melhor Acurácia no conjunto de validação: ", best_accuracy, file=txt)

        svm_best = SVC(kernel=best_params['kernel'], C=best_params['C'], random_state=42, probability=True)
        svm_best.fit(x_train, y_train)
        svm_probs = svm_best.predict_proba(x_test)

        opiniao_test = svm_best.predict(x_test)
        accuracy_test = accuracy_score(y_test, opiniao_test)
        print("Acurácia no conjunto de teste: ", accuracy_test, file=txt)
        print(accuracy_test) # vai pro csv na coluna SVM
        print("Matriz de Confusão\n ",confusion_matrix(y_test, opiniao_test), file=txt)
        TN, FP, FN, TP = confusion_matrix(y_test, opiniao_test).ravel()
        print("Sensibilidade: ",(TP/(TP+FN)), file=txt)
        print("Especificade: ",(TN/(FP+TN)), file=txt)

        accuracy.append(accuracy_test)

        #====================Start of Decision Tree Classifier================
        print("\nDecision Tree", file=txt)

        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 10]
        }

        best_accuracy = 0
        best_params = {}

        for criterion in param_grid['criterion']:
            for max_depth in param_grid['max_depth']:
                for min_samples_split in param_grid['min_samples_split']:
                    for min_samples_leaf in param_grid['min_samples_leaf']:
                        decision_tree = DecisionTreeClassifier(
                            criterion=criterion,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=42
                        )
                        decision_tree.fit(x_train, y_train)
                        
                        opiniao_val = decision_tree.predict(x_val)
                        accuracy_val = accuracy_score(y_val, opiniao_val)
                        
                        if accuracy_val > best_accuracy:
                            best_accuracy = accuracy_val
                            best_params = {
                                'criterion': criterion,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf
                            }

        print("Melhores Parâmetros: ", best_params, file=txt)
        print("Melhor Acurácia no conjunto de validação: ", best_accuracy, file=txt)

        decision_tree_best = DecisionTreeClassifier(
            criterion=best_params['criterion'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=42
        )
        decision_tree_best.fit(x_train, y_train)
        dt_probs = decision_tree_best.predict_proba(x_test)

        opiniao_test = decision_tree_best.predict(x_test)
        accuracy_test = accuracy_score(y_test, opiniao_test)
        print("Acurácia no conjunto de teste: ", accuracy_test, file=txt)
        print(accuracy_test) # vai pro csv na coluna DT
        print("Matriz de Confusão\n ",confusion_matrix(y_test, opiniao_test), file=txt)
        TN, FP, FN, TP = confusion_matrix(y_test, opiniao_test).ravel()
        print("Sensibilidade: ",(TP/(TP+FN)), file=txt)
        print("Especificade: ",(TN/(FP+TN)), file=txt)

        accuracy.append(accuracy_test)

        #====================Start of MLP Classifier================
        print("\nMLP", file=txt)
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'max_iter': [1000, 1500, 2000],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        }

        best_accuracy = 0
        best_params = {}

        for hidden_layer_sizes in param_grid['hidden_layer_sizes']:
            for activation in param_grid['activation']:
                for max_iter in param_grid['max_iter']:
                    for learning_rate in param_grid['learning_rate']:
                        mlp_model = MLPClassifier(
                            hidden_layer_sizes=hidden_layer_sizes,
                            activation=activation,
                            max_iter=max_iter,
                            learning_rate=learning_rate,
                            random_state=42
                        )
                        mlp_model.fit(x_train, y_train)
                        
                        opiniao_val = mlp_model.predict(x_val)
                        accuracy_val = accuracy_score(y_val, opiniao_val)
                        
                        if accuracy_val > best_accuracy:
                            best_accuracy = accuracy_val
                            best_params = {
                                'hidden_layer_sizes': hidden_layer_sizes,
                                'activation': activation,
                                'max_iter': max_iter,
                                'learning_rate': learning_rate
                            }

        print("Melhores Parâmetros: ", best_params, file=txt)
        print("Melhor Acurácia no conjunto de validação: ", best_accuracy, file=txt)

        mlp_best = MLPClassifier(
            hidden_layer_sizes=best_params['hidden_layer_sizes'],
            activation=best_params['activation'],
            max_iter=best_params['max_iter'],
            learning_rate=best_params['learning_rate'],
            random_state=42
        )
        mlp_best.fit(x_train, y_train)
        mlp_probs = mlp_best.predict_proba(x_test)

        opiniao_test = mlp_best.predict(x_test)
        accuracy_test = accuracy_score(y_test, opiniao_test)
        print("Acurácia no conjunto de teste: ", accuracy_test, file=txt)
        print(accuracy_test) # vai pro csv na coluna MLP
        print("Matriz de Confusão\n ",confusion_matrix(y_test, opiniao_test), file=txt)
        TN, FP, FN, TP = confusion_matrix(y_test, opiniao_test).ravel()
        print("Sensibilidade: ",(TP/(TP+FN)), file=txt)
        print("Especificade: ",(TN/(FP+TN)), file=txt)

        accuracy.append(accuracy_test)

        #====================Start of Naive Bayes Classifier================
        print("\nNaive Bayes", file=txt)

        naive_bayes = BernoulliNB()
        naive_bayes.fit(x_train, y_train)
        nb_probs = naive_bayes.predict_proba(x_test)

        opiniao_test = naive_bayes.predict(x_test)
        accuracy_test = accuracy_score(y_test, opiniao_test)
        print("Acurácia no conjunto de teste: ", accuracy_test, file=txt)
        print(accuracy_test) # vai pro csv na coluna NB
        print("Matriz de Confusão\n ",confusion_matrix(y_test, opiniao_test), file=txt)
        TN, FP, FN, TP = confusion_matrix(y_test, opiniao_test).ravel()
        print("Sensibilidade: ",(TP/(TP+FN)), file=txt)
        print("Especificade: ",(TN/(FP+TN)), file=txt)

        accuracy.append(accuracy_test)

        # Armazenando as acurácias na lista
        accuracy_list.append(accuracy)

        #====================Regra da Soma================================
        # Probabilidades de cada modelo
        knn_probs = KNN_best.predict_proba(x_test)
        svm_probs = svm_best.predict_proba(x_test)
        dt_probs = decision_tree_best.predict_proba(x_test)
        mlp_probs = mlp_best.predict_proba(x_test)
        nb_probs = naive_bayes.predict_proba(x_test)

        # Somando as probabilidades
        summed_probs = knn_probs + svm_probs + dt_probs + mlp_probs + nb_probs
        final_predictions = np.argmax(summed_probs, axis=1)

        # Avaliando o desempenho
        accuracy_test_sum_rule = accuracy_score(y_test, final_predictions)
        print("Acurácia do multiclassificador no conjunto de teste (Regra da Soma): ", accuracy_test_sum_rule, file=txt)
        print(accuracy_test_sum_rule)
        print("Matriz de Confusão\n", confusion_matrix(y_test, final_predictions), file=txt)

        #====================Voto Majoritário==========================
        # Voto Majoritário manual
        def voto_majoritario(predictions):
            # Inicializando a lista para armazenar as predições finais
            final_predictions = []

            # Transpondo as predições para iterar por amostra
            for i in range(predictions.shape[1]):
                # Extraindo a coluna i-ésima (predições para a i-ésima amostra)
                votes = predictions[:, i]
                
                # Contando a frequência de cada classe
                unique, counts = np.unique(votes, return_counts=True)
                
                # Selecionando a classe com o maior número de votos
                final_predictions.append(unique[np.argmax(counts)])
            
            return np.array(final_predictions)


        # Predições de cada modelo
        pred_knn = KNN_best.predict(x_test)
        pred_svm = svm_best.predict(x_test)
        pred_dt = decision_tree_best.predict(x_test)
        pred_mlp = mlp_best.predict(x_test)
        pred_nb = naive_bayes.predict(x_test)

        # Empilhando as predições
        predictions = np.array([pred_knn, pred_svm, pred_dt, pred_mlp, pred_nb])

        # Voto Majoritário
        opiniao_test = voto_majoritario(predictions)
        

        accuracy_test_majority_vote = accuracy_score(y_test, opiniao_test)
        print("Acurácia final no conjunto de teste (Voto Majoritário): ", accuracy_test_majority_vote, file=txt)
        print(accuracy_test_majority_vote)
        print("Matriz de Confusão\n", confusion_matrix(y_test, opiniao_test), file=txt)

        # =================Borda Count==================================
        # Função para cálculo do Borda Count
        def borda_count(predictions, classes):
            borda_scores = np.zeros((predictions.shape[1], len(classes)))

            for i, pred in enumerate(predictions):
                for j, class_label in enumerate(classes):
                    borda_scores[:, j] += (pred == class_label).astype(int) * (predictions.shape[0] - i)
            
            final_predictions = classes[np.argmax(borda_scores, axis=1)]
            return final_predictions
        
        # Cálculo do Borda Count
        opiniao_test_borda = borda_count(predictions, np.unique(y_test))

        accuracy_test_borda = accuracy_score(y_test, opiniao_test_borda)
        print("Acurácia final no conjunto de teste (Borda Count): ", accuracy_test_borda, file=txt)
        print(accuracy_test_borda)
        print("Matriz de Confusão\n", confusion_matrix(y_test, opiniao_test_borda), file=txt)

        # Salvando as acurácias em uma lista
        accuracy_list_mult.append([
            accuracy_test_sum_rule, 
            accuracy_test_majority_vote, 
            accuracy_test_borda
        ])

# Criando o DataFrame e salvando no arquivo CSV
df = pd.DataFrame(accuracy_list, columns=['KNN', 'SVM', 'DT', 'MLP','NB'])
df.to_csv('results.csv', index=False)
df_means = df.mean()

df_mult = pd.DataFrame(accuracy_list_mult, columns=['Regra da Soma', 'Voto Majoritário', 'Borda Count'])
df_mult.to_csv('results_multi.csv', index=False)

#========================= ESTATISTICA CLASSIFIDORES "MONO" ==============================

knn_mean = df_means['KNN']
svm_mean = df_means['SVM']
dt_mean = df_means['DT']
mlp_mean = df_means['MLP']
nb_mean = df_means['NB']


# Aplicando o teste de Kruskal-Wallis
stat, p_value = stats.kruskal([knn_mean], [svm_mean], [dt_mean], [mlp_mean], [nb_mean])

# Exibindo o resultado
print(f'Estatística de Kruskal-Wallis: {stat}')
print(f'Valor-p: {p_value}')

# Interpretando o resultado
alpha = 0.05
if p_value < alpha:
    print("Há diferença estatisticamente significativa entre os classificadores.")
else:
    print("Não há diferença estatisticamente significativa entre os classificadores.")

# Obtenha os nomes das colunas (classificadores)
classifiers = df.columns

# Comparando de 2 em 2 com o teste de Mann-Whitney
comparisons = {}
for i in range(len(classifiers)):
    for j in range(i + 1, len(classifiers)):
        clf1, clf2 = classifiers[i], classifiers[j]
        comparisons[f'{clf1} vs {clf2}'] = mannwhitneyu(df[clf1], df[clf2])

# Exibindo os resultados com interpretação
significance_level = 0.05

for comparison, result in comparisons.items():
    print(f'{comparison}: U={result.statistic}, p-value={result.pvalue}')
    if result.pvalue < significance_level:
        print(f"Resultado: Existe uma diferença estatisticamente significativa entre {comparison}.")
    else:
        print(f"Resultado: Não há diferença estatisticamente significativa entre {comparison}.")

#=============================== ESTATISTICA MULTI CLASSIFICADORES ==============================

# Supondo que accuracy_list_mult já foi criado e contém as acurácias dos multi-classificadores
# Criando o DataFrame e salvando no arquivo CSV
df_mult = pd.DataFrame(accuracy_list_mult, columns=['Regra da Soma', 'Voto Majoritário', 'Borda Count'])
df_mult.to_csv('results_multi.csv', index=False)
df_mult_means = df_mult.mean()

# Extraindo as médias para cada classificador
soma_mean = df_mult_means['Regra da Soma']
voto_mean = df_mult_means['Voto Majoritário']
borda_mean = df_mult_means['Borda Count']

# Aplicando o teste de Kruskal-Wallis para os multi-classificadores
stat, p_value = stats.kruskal(df_mult['Regra da Soma'], df_mult['Voto Majoritário'], df_mult['Borda Count'])

# Exibindo o resultado de Kruskal-Wallis
print(f'Estatística de Kruskal-Wallis: {stat}')
print(f'Valor-p: {p_value}')

# Interpretando o resultado
alpha = 0.05
if p_value < alpha:
    print("Há diferença estatisticamente significativa entre os multi-classificadores.")
else:
    print("Não há diferença estatisticamente significativa entre os multi-classificadores.")

# Obtenha os nomes das colunas (multi-classificadores)
mult_classifiers = df_mult.columns

# Comparando de 2 em 2 com o teste de Mann-Whitney
mult_comparisons = {}
for i in range(len(mult_classifiers)):
    for j in range(i + 1, len(mult_classifiers)):
        clf1, clf2 = mult_classifiers[i], mult_classifiers[j]
        mult_comparisons[f'{clf1} vs {clf2}'] = mannwhitneyu(df_mult[clf1], df_mult[clf2])

# Exibindo os resultados com interpretação
for comparison, result in mult_comparisons.items():
    print(f'{comparison}: U={result.statistic}, p-value={result.pvalue}')
    if result.pvalue < alpha:
        print(f"Resultado: Existe uma diferença estatisticamente significativa entre {comparison}.")
    else:
        print(f"Resultado: Não há diferença estatisticamente significativa entre {comparison}.")

