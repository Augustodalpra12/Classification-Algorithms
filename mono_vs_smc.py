import pandas as pd
from scipy.stats import mannwhitneyu

# Abrindo os CSVs
df = pd.read_csv('results.csv')
df_multi = pd.read_csv('results_multi.csv')

# Selecionando as colunas desejadas
df_new = df[['KNN', 'SVM', 'DT', 'MLP', 'NB']]
df_multi = df_multi[['Regra da Soma', 'Voto Majoritário', 'Borda Count']]

# Calculando a média de cada coluna e armazenando em variáveis
knn_mean = df_new['KNN'].mean()
svm_mean = df_new['SVM'].mean()
dt_mean = df_new['DT'].mean()
mlp_mean = df_new['MLP'].mean()
nb_mean = df_new['NB'].mean()

regra_soma_mean = df_multi['Regra da Soma'].mean()
voto_majoritario_mean = df_multi['Voto Majoritário'].mean()
borda_count_mean = df_multi['Borda Count'].mean()

# Exibindo as médias
print("Médias dos classificadores:")
print(f"KNN: {knn_mean}")
print(f"SVM: {svm_mean}")
print(f"DT: {dt_mean}")
print(f"MLP: {mlp_mean}")
print(f"NB: {nb_mean}")

print("\nMédias dos métodos de combinação:")
print(f"Regra da Soma: {regra_soma_mean}")
print(f"Voto Majoritário: {voto_majoritario_mean}")
print(f"Borda Count: {borda_count_mean}")

# SVM x Voto Majoritário usando Mann-Whitney
# Comparando os valores das duas colunas originais
stat, p_value = mannwhitneyu(svm_mean, voto_majoritario_mean)

# Exibindo o resultado
print(f'\nEstatística de Mann-Whitney: {stat}')
print(f'Valor-p: {p_value}')

# Interpretando o resultado
alpha = 0.05
if p_value < alpha:
    print("Há diferença estatisticamente significativa entre os classificadores.")
else:
    print("Não há diferença estatisticamente significativa entre os classificadores.")
