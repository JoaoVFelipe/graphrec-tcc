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

USER_NUM=3000
ITEM_NUM=10000

####### FUNÇÕES CITEULIKE ####### 
def get_citeulike_raw():
    colnames=['doc.id', 'title', 'citeulike.id', 'raw.title', 'raw.abstract'] 
    return pd.read_csv('./data/citeulike/raw-data.csv', header=None, encoding='ISO-8859-1',  skiprows = 1, names=colnames)
     
def get_citations():
    return pd.read_csv('./data/citeulike/citations.dat', header=None)

def get_item_tag():
     return pd.read_csv('./data/citeulike/item-tag.dat', header=None)

def get_tags():
    return pd.read_csv('./data/citeulike/tags.dat', header=None)

def get_ItemDataCiteULike(df_test):
    print("Getting raw data...")
    df = get_citeulike_raw()
    df = df.drop('raw.title', axis=1)
    df = df.drop('citeulike.id', axis=1)
    df['doc.id'] = df['doc.id'].apply(lambda x: x - 1)
    df = df.rename(columns={'raw.abstract': 'abstract'})
    df_reduced = pd.DataFrame(columns=df.keys().values.tolist()) 

    print("Reducing to relevant data...")
    for doc_id in df_test['item'].unique():
        df_reduced.loc[df.index[doc_id]] = df.iloc[doc_id]

    print("Cleaning data...")
    del df
    df_reduced = df_reduced.drop('title', axis=1)
    df_reduced = df_reduced.drop('abstract', axis=1)

    print("Getting tags and tags description...")
    df_item_tags = get_item_tag()
    tags = get_tags()

    print("Processing new dataset...")
    ## Add all tags as columns and set 0 and 1 for each
    for index, row in df_reduced.iterrows():
        item_tags = list(map(int, df_item_tags[0][row['doc.id']].split()))
        item_tags.pop(0)
        for tag_id in item_tags:
            tag_name = tags[0][tag_id]
            if tag_name not in df_reduced:
                df_reduced[tag_name] = 0
            ### Ao invés de alterar no index, alterar no doc id
            df_reduced.at[index, tag_name] = 1
        del item_tags

    df_reduced.drop('doc.id', axis=1)
    print("Processing complete")
    print("Cleaning data...")
    del df_item_tags
    del tags
    print("Sorting and filling indexes...")
    df_reduced = df_reduced.sort_index()
    df_reduced = df_reduced.reindex(range(df_reduced.index[0], df_reduced.index[-1] + 1), fill_value=0)
    return df_reduced.values

def get_data_citeulike(dataset, perc=0.9):
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
    print("Reduzindo dimensionalidade")
    print("Tamanho inicial --- ", df_ratings.size)
    index_max_user = df_ratings.loc[df_ratings.user==MAX_USERS_DESIRED].index[0]
    df_ratings = df_ratings[:index_max_user]
    df_ratings.drop(df_ratings[df_ratings['item'] >= MAX_ITEMS_DESIRED].index, inplace = True)
    print("Redução completa")
    print("Usuários restantes --- ", df_ratings['user'].max())
    print("Items restantes --- ", df_ratings['item'].max())
    print("Tamanho final da base --- ", df_ratings.size)
    return df_ratings

  
def read_process(filname, sep="\t"):
    col_names = ["user", "item", "rate", "st"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    df["user"] -= 1
    df["item"] -= 1
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    return df

def pre_process_dataset():
    print("Iniciando script...")
    print("Carregando dataset principal...")
    df_user_item = pd.read_csv('./data/citeulike/users.dat', header=None)
    ### Transformar essa lista em uma matriz de conexão
    print("Criando matriz de conexão...")

    rows_list = []
    for ind in df_user_item.index:
        user_id = ind
        user_ratings = list(map(int, df_user_item[0][ind].split()))
        for rating in user_ratings:
            user_rating = [user_id, rating, 1]
            rows_list.append(user_rating)
    df_ratings = pd.DataFrame(columns=['user', 'item', 'rate'], data=rows_list)

    ### Redução de dimensionalidade, se desejado
    df_ratings_sample = df_ratings
    del df_ratings

    docs = df_ratings_sample['item'].unique()
    users = df_ratings_sample['user'].unique()
    rows_list = []
    ##df_ratings_complete = df_ratings_complete.iloc[0:0]

    print("Usuário e documentos finais:", users.size, docs.size)

    ### Dataset completo
    print("Processamento completo. Criando dataframe com ", len(rows_list), " items.")
    ### Substitui os valores de 0 e 1 para valores mais compreensiveis pelo graphrec
    # df_ratings_complete = pd.DataFrame(columns=['user', 'item', 'rate'], data=rows_list)
    df_ratings_complete = df_ratings_sample
    # df_ratings_complete.loc[df_ratings_complete['rate'] == 1, 'rate'] = POSITIVE_VALUE
    # df_ratings_complete.loc[df_ratings_complete['rate'] == 0, 'rate'] = NEGATIVE_VALUE
    # random_sampling = df_ratings_complete.groupby("rate").sample(n=14000, random_state=42)
    print("Limpando dados...")
    del rows_list
    del docs
    del users
    del df_ratings_sample
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
                    df_train, df_test = get_data_citeulike(df_ratings_complete)

                    model = GraphRec(df_train, df_test, ItemData=False, UserData = False, Graph=False, Dataset='citeU', USER_NUM=USER_NUM, ITEM_NUM=ITEM_NUM, 
                            MFSIZE=mf_size,
                            UW=ur,
                            IW=ir,
                            LR=lr,
                            EPOCH_MAX = 200,
                            BATCH_SIZE = 500) 
                    
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
    return  best_lr, best_ur, best_ir, best_mfsize
    
### Check if pre processed dataset is already saved
preprocessed_path = 'preprocess/citeulike_pre_processed_dataset-1000:5000.csv'
file_exists = exists(preprocessed_path)

if(not file_exists):
    df_ratings_complete = pre_process_dataset()
    df_ratings_complete.to_csv(preprocessed_path)
else:
    df_ratings_complete = pd.read_csv(preprocessed_path)

### Get best parameters
#best_lr, best_ur, best_ir, best_mfsize = grid_search()
### Best parameters based on Grid Search
best_lr = 0.001 
best_ur = 0.01 
best_ir = 0.1 
best_mfsize = 100

#### Primeira execução sem informações dos artigos:
df_train, df_test = get_data_citeulike(df_ratings_complete)
docs = df_ratings_complete['item'].unique()

### Adiciona opções negativas na base de treino
NEGATIVE_RATIO = 50

negative_ratings = []
for user in df_train['user'].unique():
    # Itens avaliados pelo usuário
    user_items = df_train[df_train['user'] == user]['item'].values
    # Itens que não foram avaliados pelo usuário
    non_user_items = list(set(docs).difference(user_items))
    
    # Para cada rating positivo do usuário
    for item in user_items:
        # Selecionar x itens não avaliados aleatoriamente
        sampled_negative_items = np.random.choice(non_user_items, size=NEGATIVE_RATIO, replace=False)
        for neg_item in list(sampled_negative_items):
            negative_ratings.append({'user': user, 'item': neg_item, 'rate': 0})

df_negative = pd.DataFrame(negative_ratings)
# Combinar os ratings positivos e negativos
df_train_final = pd.concat([df_train, df_negative]).reset_index(drop=True)
df_train_final = shuffle(df_train_final).reset_index(drop=True)

### Adiciona uma opção negativa para cada positiva na base de testes: 
test_users = df_test['user'].unique()

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
        negative_ratings.append({'user': user, 'item': list(sampled_negative_items)[0], 'rate': 0})

df_negative = pd.DataFrame(negative_ratings)
# Combinar os ratings positivos e negativos
df_test_final = pd.concat([df_test, df_negative]).reset_index(drop=True)
# Embaralhar o dataframe final
df_test_final = shuffle(df_test_final).reset_index(drop=True)


print("------------- Execução com os melhores parâmetros: Sem informações adicionais dos artigos: ---------------")
model = GraphRec(df_train_final, df_test_final, ItemData=False, UserData = False, Graph=False, Dataset='citeU', USER_NUM=USER_NUM, ITEM_NUM=ITEM_NUM, 
            MFSIZE=best_mfsize,
            UW=best_ur,
            IW=best_ir,
            LR=best_lr,
            EPOCH_MAX = 200,
            BATCH_SIZE = 500) 
print("Execução da predição para teste")
df_test_final = df_test_final.sort_values('user') # retorna o dataset de treino ordenado com base nos ids
qids = df_test_final['user'] # qids sao os ids dos usuarios
y_test = df_test_final['rate'] # y_test sao os scores verdadeiros do teste
predictions = np.array(model.predict(df_test_final)).flatten() # model.predict(df_test)

result_df = df_test_final.copy(deep=True)
result_df = result_df.assign(prediction=predictions)

print("Resultados Primeira Execução: ")
ndcgs = queries_ndcg(y_test, predictions, qids) # retorna uma lista com ndcg de cada query (que seria id de cada usuario)
print("MEAN NDCGS:", ndcgs.mean())
rmse = mean_squared_error(y_test, predictions, squared = False) # retorna a raiz quadradica do erro-medio
print("RMSE:", rmse)
map_res = mean_ap(result_df, 1.0)
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
