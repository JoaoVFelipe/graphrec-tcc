import numpy as np
import pandas as pd
import random
from os.path import exists

from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

from utils.graphrec_automated import GraphRec
from utils.metrics import queries_ndcg, mean_ap

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings('ignore')


POSITIVE_VALUE = 5
NEGATIVE_VALUE = 1

USER_NUM=5000
ITEM_NUM=15000

def get_dataset(dataset, perc=0.9):
    rows = len(dataset)
    df = dataset.iloc[np.random.permutation(rows)].reset_index(drop=True)
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
        df["rate"] = df["rate"].astype(np.float32)

    split_index = int(rows * perc)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test

def reduce_dimensionality(df_ratings, MAX_USERS_DESIRED = USER_NUM, MAX_ITEMS_DESIRED = ITEM_NUM):
    # print("Reduzindo dimensionalidade")
    # print("Tamanho inicial --- ", df_ratings.size)
    # index_max_user = df_ratings.loc[df_ratings.user >= MAX_USERS_DESIRED].index[0]
    # df_ratings = df_ratings[:index_max_user]
    # df_ratings.drop(df_ratings[df_ratings['item'] >= MAX_ITEMS_DESIRED].index, inplace = True)
    # print("Redução completa")
    # print("Usuários restantes --- ", df_ratings['user'].max())
    # print("Items restantes --- ", df_ratings['item'].max())
    # print("Tamanho final da base --- ", df_ratings.size)
    # return df_ratings
    # Filtra o DataFrame original para manter apenas as linhas correspondentes aos N usuários mais frequentes
    df_filtrado = df_ratings[(df_ratings['user'] <= MAX_USERS_DESIRED) & (df_ratings['item'] <= MAX_ITEMS_DESIRED)]
    
    return df_filtrado

  
def read_process(filname, sep="\t"):
    col_names = ["user", "item", "rate", "st"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    df["user"] -= 1
    df["item"] -= 1
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    return df

def replace_value(value):
    if value > 3:
        return 1
    else:
        return 0

def pre_process_dataset():
    print("Iniciando script...")
    print("Carregando dataset principal...")
    df_user_item = pd.read_csv('./data/ifgproduz/relation_list.txt',  sep="\t")
    ### Transformar essa lista em uma matriz de conexão
    print("Criando matriz de conexão...")

    df_user_item.rename(columns={'nota': 'rate'}, inplace=True)
    df_user_item = df_user_item.rename({'id_producao_id': 'item'}, axis=1)
    df_user_item = df_user_item.rename({'id_curriculo_id': 'user'}, axis=1)

        # usuario_map = {usuario_antigo: novo_id for novo_id, usuario_antigo in enumerate(df_user_item['user'].unique())}
    # df_user_item['org_user'] = df_user_item['user']  # Adiciona uma coluna para o ID de usuário antigo
    # df_user_item['user'] = df_user_item['user'].map(usuario_map)

    users = df_user_item['user'].unique()
    users.sort()
    new_id = 0
    for u in users:
        df_user_item.loc[df_user_item['user'] == u, 'org_user'] = u
        df_user_item.loc[df_user_item['user'] == u, 'user'] = new_id
        new_id = new_id + 1

    # Reajusta os IDs de itens
    # item_map = {item_antigo: novo_id for novo_id, item_antigo in enumerate(df_user_item['item'].unique())}
    # df_user_item['org_item'] = df_user_item['item']  # Adiciona uma coluna para o ID de item antigo
    # df_user_item['item'] = df_user_item['item'].map(item_map)

    items = df_user_item['item'].unique()
    items.sort()
    new_id = 0
    for i in items:
        df_user_item.loc[df_user_item['item'] == i, 'org_item'] = i
        df_user_item.loc[df_user_item['item'] == i, 'item'] = new_id
        new_id = new_id + 1
    
    df_user_item['rate'] = df_user_item['rate'].clip(0.0, 5.0)

    ### Redução de dimensionalidade, se desejado
    ### df_ratings_complete = reduce_dimensionality(df_user_item, len(users), len(items))
    df_ratings_complete = df_user_item
    return df_ratings_complete

def grid_search():
    learning_rates = [0.001, 0.00005, 0.000005]
    user_regularizers = [0.1, 0.05, 0.01]
    item_regularizers = [0.1, 0.05,  0.01]
    mf_sizes = [2, 30, 100]

    best_rmse = 100

    best_lr = 0
    best_ur = 0
    best_ir = 0
    best_mfsize = 0

    for lr in learning_rates:
        for ur in user_regularizers:
            for ir in item_regularizers:
                for mf_size in mf_sizes:
                    df_train, df_test = get_dataset(df_ratings_complete)

                    model = GraphRec(df_train, df_test, ItemData=False, UserData = False, Graph=False, Dataset='DBLP', USER_NUM=USER_NUM, ITEM_NUM=ITEM_NUM, 
                            MFSIZE=mf_size,
                            UW=ur,
                            IW=ir,
                            LR=lr,
                            EPOCH_MAX = 200,
                            BATCH_SIZE = 1000) 
                    
                    df_test = df_test.sort_values('user') # retorna o dataset de treino ordenado com base nos ids
                    y_test = df_test['rate']                    
                    predictions = np.array(model.predict(df_test)).flatten() # model.predict(df_test)
                    rmse = mean_squared_error(y_test, predictions, squared = False) # retorna a raiz quadradica do erro-medio

                    if(rmse < best_rmse):
                        best_lr = lr
                        best_ur = ur
                        best_ir = ir
                        best_mfsize = mf_size

                        best_rmse = rmse
                        print("Parâmetros melhores encontrados: ")
                        print("LR --- ", best_lr, "|| UR --- ", best_ur, "|| IR --- ", best_ir, "|| MF_SIZE --- ", best_mfsize, "|| RMSE --- ", best_rmse)
                    else:
                        print("RMSE inferior ao melhor ---", rmse)
                    del model
    return  best_lr, best_ur, best_ir, best_mfsize
    
### Check if pre processed dataset is already saved
preprocessed_path = 'preprocess/ifgproduz_pre_processed_dataset-1000:5000.csv'
file_exists = exists(preprocessed_path)

if(not file_exists):
    df_ratings_complete = pre_process_dataset()
    df_ratings_complete.to_csv(preprocessed_path)
else:
    df_ratings_complete = pd.read_csv(preprocessed_path)

### Get best parameters
##best_lr, best_ur, best_ir, best_mfsize = grid_search()
### Best parameters based on Grid Search
best_lr = 0.0002
best_ur = 0.08
best_ir = 0.02
best_mfsize = 50

#### Primeira execução sem informações dos artigos:
df_train, df_test = get_dataset(df_ratings_complete)

users = df_ratings_complete['user'].unique()
items = df_ratings_complete['item'].unique()

# ### Adiciona opções negativas na base de treino
docs = df_ratings_complete['item'].unique()
# NEGATIVE_RATIO = 50

# negative_ratings = []
# for user in df_train['user'].unique():
#     # Itens avaliados pelo usuário
#     user_items = df_train[df_train['user'] == user]['item'].values
#     # Itens que não foram avaliados pelo usuário
#     non_user_items = list(set(docs).difference(user_items))
    
#     # Para cada rating positivo do usuário
#     for item in user_items:
#         # Selecionar x itens não avaliados aleatoriamente
#         sampled_negative_items = np.random.choice(non_user_items, size=NEGATIVE_RATIO, replace=False)
#         for neg_item in list(sampled_negative_items):
#             negative_ratings.append({'user': user, 'item': neg_item, 'rate': 0})

# df_negative = pd.DataFrame(negative_ratings)
# # Combinar os ratings positivos e negativos
# df_train_final = pd.concat([df_train, df_negative]).reset_index(drop=True)
# df_train_final = shuffle(df_train_final).reset_index(drop=True)

# ### Adiciona uma opção negativa para cada positiva na base de testes: 
# test_users = df_test['user'].unique()

negative_ratings = []
print("Adicionando itens não explicitos a matriz de testes...")
### Cria todas as conexões restantes não apresentadas no dataset inicial - Rate == 0
# NEGATIVE_RATIO = 50
for user in df_test['user'].unique():
    # Itens avaliados pelo usuário
    user_items = df_test[df_test['user'] == user]['item'].values
    # Itens que não foram avaliados pelo usuário
    non_user_items = list(set(docs).difference(user_items))
    
    # Para cada rating positivo do usuário
    for item in user_items:
        # Selecionar x itens não avaliados aleatoriamente
        sampled_negative_items = np.random.choice(non_user_items, size=1, replace=False)
        negative_ratings.append({'user': user, 'item': list(sampled_negative_items)[0], 'rate': 0, 'org_user': '', 'org_item': ''})

df_negative = pd.DataFrame(negative_ratings)
# Combinar os ratings positivos e negativos
df_test_final = pd.concat([df_test, df_negative]).reset_index(drop=True)
# Embaralhar o dataframe final
df_test_final = shuffle(df_test_final).reset_index(drop=True)

print("------------- Execução com os melhores parâmetros: Sem informações adicionais dos artigos: ---------------")
model = GraphRec(df_train, df_test_final, ItemData=False, UserData = False, Graph=True, Dataset='IFG', USER_NUM=len(users), ITEM_NUM=len(items), 
            MFSIZE=best_mfsize,
            UW=best_ur,
            IW=best_ir,
            LR=best_lr,
            EPOCH_MAX = 550,
            BATCH_SIZE = 1000,
            orig_graphrec=True) 
print("Execução da predição para teste")
df_test_final = df_test.sort_values('user') # retorna o dataset de treino ordenado com base nos ids
qids = df_test_final['user'] # qids sao os ids dos usuarios
y_test = df_test_final['rate'] # y_test sao os scores verdadeiros do teste
predictions = np.array(model.predict(df_test_final)).flatten() # model.predict(df_test)

result_df = df_test_final.copy(deep=True)
result_df = result_df.assign(prediction=predictions)

print("Resultados Primeira Execução: ")
ndcgs = queries_ndcg(y_test, predictions, qids, 10) # retorna uma lista com ndcg de cada query (que seria id de cada usuario)
print("MEAN NDCGS:", ndcgs.mean())
rmse = mean_squared_error(y_test, predictions, squared = False) # retorna a raiz quadradica do erro-medio
print("RMSE:", rmse)
map_res = mean_ap(result_df, 3.0)
print("MAP:", map_res)
print("------------- Execução finalizada ---------------")


#### Segunda execução: Com informações dos artigos:
# print("------------- Segunda execução com os melhores parâmetros: Com informações adicionais dos artigos: ---------------")
# df_train, df_test = get_data_citeulike(df_ratings_complete)
# model = GraphRec(df_train, df_test, ItemData=True, UserData = False, Graph=True, Dataset='citeU', USER_NUM=USER_NUM, ITEM_NUM=ITEM_NUM)
# print("Execução da predição para teste")
# df_test = df_test.sort_values('user') # retorna o dataset de treino ordenado com base nos ids
# qids = df_test['user'] # qids sao os ids dos usuarios
# y_test = df_test['rate'] # y_test sao os scores verdadeiros do teste
# predictions = np.array(model.predict(df_test)).flatten() # model.predict(df_test) 

# result_df = df_test.copy(deep=True)
# result_df = result_df.assign(prediction=predictions)

# print("Resultados Segunda Execução: ")
# ndcgs = queries_ndcg(y_test, predictions, qids) # retorna uma lista com ndcg de cada query (que seria id de cada usuario)
# print("MEAN NDCGS:", ndcgs.mean())
# rmse = mean_squared_error(y_test, predictions, squared = False) # retorna a raiz quadradica do erro-medio
# print("RMSE:", rmse)
# map_res = mean_ap(result_df, 5.0)
# print("MAP:", map_res)
# print("------------- Execução finalizada ---------------")
