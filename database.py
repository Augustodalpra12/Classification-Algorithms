import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# dataset
# https://www.kaggle.com/datasets/salikhussaini49/nba-dataset

# Carregar os arquivos CSV
player_game_stats_file = 'data/Player_Game_Stats_1949_2024.csv'
nba_players_file = 'data/NBA_PLAYERS.csv'

player_game_stats_df = pd.read_csv(player_game_stats_file)
nba_players_df = pd.read_csv(nba_players_file)

# Converter a coluna "GAME_DATE" para o formato de data
player_game_stats_df['GAME_DATE'] = pd.to_datetime(player_game_stats_df['GAME_DATE'])

# Filtrar as linhas onde "GAME_DATE" é maior que 21 de dezembro de 2020
filtered_df = player_game_stats_df[player_game_stats_df['GAME_DATE'] > '2020-12-21']

# Selecionar as colunas de interesse
filtered_df = filtered_df[['Player_ID', 'SEASON_ID', 'FG_PCT', 'FG3_PCT', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PTS', 'PLUS_MINUS']]

# Obter combinações únicas de Player_ID e SEASON_ID
unique_combinations = filtered_df[['Player_ID', 'SEASON_ID']].drop_duplicates()

# Filtrar o DataFrame de jogadores com base nos Player_IDs únicos
filtered_nba_players_df = nba_players_df[nba_players_df['id'].isin(unique_combinations['Player_ID'])]

# Mesclar as informações dos jogadores com as combinações únicas de Player_ID e SEASON_ID
merged_df = unique_combinations.merge(filtered_nba_players_df, left_on='Player_ID', right_on='id', how='left')

# Mesclar com as estatísticas do jogo
final_df = merged_df.merge(filtered_df, on=['Player_ID', 'SEASON_ID'], how='left')

# Selecionar e renomear as colunas de interesse
final_df = final_df[['Player_ID', 'SEASON_ID', 'full_name', 'FG_PCT', 'FG3_PCT', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PTS', 'PLUS_MINUS']]

# Remover as linhas que contêm valores NaN
final_df = final_df.dropna()

# Adicionar a coluna "games_played" contando o número de aparições de cada jogador
final_df['games_played'] = final_df.groupby(['Player_ID', 'SEASON_ID'])['Player_ID'].transform('count')

# Filtrar os jogadores que jogaram pelo menos 41 jogos
final_df = final_df[final_df['games_played'] >= 41]

# Listas dos jogadores selecionados como All-NBA por temporada, em minúsculas
all_nba_2020 = [
    'giannis antetokounmpo', 'kawhi leonard', 'nikola jokic', 'stephen curry', 'luka doncic', 'julius randle', 'lebron james', 'joel embiid', 'chris paul', 'damian lillard', 'jimmy butler', 'paul george', 'rudy gobert', 'bradley beal', 'kyrie irving'
]

all_nba_2021 = [
    'giannis antetokounmpo', 'jayson tatum', 'nikola jokic', 'devin booker', 'luka doncic',
    'demar derozan', 'kevin durant', 'joel embiid', 'ja morant', 'stephen curry',
    'pascal siakam', 'lebron james', 'karl-anthony towns', 'chris paul', 'trae young'
]
all_nba_2022 = [
    'giannis antetokounmpo', 'jayson tatum', 'joel embiid', 'shai gilgeous-alexander', 'luka doncic',
    'jimmy butler', 'jaylen brown', 'nikola jokic', 'donovan mitchell', 'stephen curry',
    'julius randle', 'lebron james', 'domantas sabonis', 'de’aaron fox', 'damian lillard'
]
all_nba_2023 = [
    'giannis antetokounmpo', 'luka doncic', 'shai gilgeous-alexander', 'nikola jokic', 'jayson tatum',
    'jalen brunson', 'anthony davis', 'kevin durant', 'anthony edwards', 'kawhi leonard',
    'devin booker', 'stephen curry', 'tyrese haliburton', 'lebron james', 'domantas sabonis'
]

# Função para determinar se o jogador é All-NBA em uma temporada específica
def is_all_nba(row):
    full_name = str(row['full_name']).strip().lower()
    if row['SEASON_ID'] == 22020:
        return 1 if full_name in all_nba_2020 else 0
    elif row['SEASON_ID'] == 22021:
        return 1 if full_name in all_nba_2021 else 0
    elif row['SEASON_ID'] == 22022:
        return 1 if full_name in all_nba_2022 else 0
    elif row['SEASON_ID'] == 22023:
        return 1 if full_name in all_nba_2023 else 0
    return 0

# Adicionar a coluna "all_nba"
final_df['all_nba'] = final_df.apply(is_all_nba, axis=1)

# Mostrar os jogadores e seus nomes para verificar comparações
for season_id in [22020, 22021, 22022, 22023]:
    season_players = final_df[final_df['SEASON_ID'] == season_id]['full_name'].unique()
    print(f'\nJogadores na temporada {season_id}:')
    print(season_players)
    
    if season_id == 22020:
        print(f'\nJogadores All-NBA 2020:')
        print(all_nba_2020)
    elif season_id == 22021:
        print(f'\nJogadores All-NBA 2021:')
        print(all_nba_2021)
    elif season_id == 22022:
        print(f'\nJogadores All-NBA 2022:')
        print(all_nba_2022)
    elif season_id == 22023:
        print(f'\nJogadores All-NBA 2023:')
        print(all_nba_2023)

# Separar em DataFrames pela SEASON_ID
season_dfs = {season_id: df for season_id, df in final_df.groupby('SEASON_ID')}

# Para cada DataFrame separado, calcular as médias das estatísticas dos jogadores e normalizar
season_avg_dfs = {}
scaler = MinMaxScaler()
for season_id, df in season_dfs.items():
    # Calcular as médias das estatísticas dos jogadores
    avg_df = df.groupby(['Player_ID', 'SEASON_ID', 'full_name', 'games_played', 'all_nba']).mean().reset_index()
    
    # Normalizar as colunas numéricas (exceto 'all_nba')
    columns_to_normalize = ['FG_PCT', 'FG3_PCT', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PTS', 'PLUS_MINUS']
    avg_df[columns_to_normalize] = scaler.fit_transform(avg_df[columns_to_normalize])
    
    # Armazenar o DataFrame normalizado
    season_avg_dfs[season_id] = avg_df
    
    # Exportar cada DataFrame para um arquivo CSV
    avg_df.to_csv(f'NBA_Player_Averages_SEASON_{season_id}.csv', index=False)

# Concatenar os DataFrames de médias normalizados em um único DataFrame
combined_df = pd.concat(season_avg_dfs.values())

# Exportar o DataFrame combinado para um único arquivo CSV
combined_df.to_csv('database.csv', index=False)

print('Arquivos CSV combinados em "database.csv" com sucesso!')
