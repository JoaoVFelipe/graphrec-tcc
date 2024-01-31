import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_squared_error
from utils.graphrec import GraphRec, get_data100k
from utils.metrics import queries_ndcg

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings('ignore')

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

    test_p = len(df_test[df_test["rate"]==1])
    test_n = len(df_test[df_test["rate"]==0])

    train_p = len(df_train[df_train["rate"]==1])
    train_n = len(df_train[df_train["rate"]==0])

    print("ITENS POSITIVOS EM TESTE: ", test_p)
    print("ITENS POSITIVOS EM TREINO: ", train_p)

    print("ITENS NEGATIVOS EM TESTE: ", test_n)
    print("ITENS NEGATIVOS EM TREINO: ", train_n)

    return df_train, df_test

def reduce_dimensionality(df_ratings, MAX_USERS_DESIRED = 1000, MAX_ITEMS_DESIRED = 5000):
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

def get_data100k():
    df = read_process("data/ml100k/u.data", sep="\t")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test

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
df_ratings = reduce_dimensionality(df_ratings, 1000, 5000)
df_ratings_sample = df_ratings
del df_ratings

docs = df_ratings_sample['item'].unique()
users = df_ratings_sample['user'].unique()
rows_list = []
##df_ratings_complete = df_ratings_complete.iloc[0:0]

print("Usuário e documentos finais:", users.size, docs.size)
count = 0

print("Adicionando itens não explicitos a matriz...")
### Cria todas as conexões restantes não apresentadas no dataset inicial - Rate == 0

NEGATIVE_RATIO = 10
for ind in users:
    neg_counter = 0
    user_id = ind
    user_ratings = list(map(int, df_user_item[0][ind].split()))
    # print("Process nr. ", count," ### User -- ", ind, "with", len(user_ratings), "items")
    for doc_id in random.shufle(docs):
        if(doc_id in user_ratings):
            user_rating = [user_id, doc_id, 1]
            rows_list.append(user_rating)
        elif neg_counter <= NEGATIVE_RATIO:
            user_rating = [user_id, doc_id, 0]
            rows_list.append(user_rating)
            neg_counter = neg_counter+1
        else:
            break
    count = count+1

### Dataset completo
print("Processamento completo. Criando dataframe com ", len(rows_list), " items.")
df_ratings_complete = pd.DataFrame(columns=['user', 'item', 'rate'], data=rows_list)
df_ratings_complete.loc[df_ratings_complete['rate'] == 1, 'rate'] = 2
df_ratings_complete.loc[df_ratings_complete['rate'] == 0, 'rate'] = 1

# random_sampling = df_ratings_complete.groupby("rate").sample(n=14000, random_state=42)

print("Limpando dados...")
del rows_list
del docs
del users
del df_ratings_sample

#### Primeira execução sem informações dos artigos:
print("------------- Primeira execução: Sem informações adicionais dos artigos: ---------------")
df_train, df_test = get_data_citeulike(df_ratings_complete)
model = GraphRec(df_train, df_test, ItemData=False, UserData = False, Graph=False, Dataset='citeU') 
print("------------- Execução finalizada ---------------")
print("Execução da predição para teste")
df_test = df_test.sort_values('user') # retorna o dataset de treino ordenado com base nos ids
qids = df_test['user'] # qids sao os ids dos usuarios
y_test = df_test['rate'] # y_test sao os scores verdadeiros do teste
predictions = np.array(model.predict(df_test)).flatten() # model.predict(df_test) 
print("Primeiras 10 predições: " , predictions[:10])

print("Resultados Primeira Execução: ")
ndcgs = queries_ndcg(y_test, predictions, qids) # retorna uma lista com ndcg de cada query (que seria id de cada usuario)
print("MEAN NDCGS:", ndcgs.mean())
rmse = mean_squared_error(y_test, predictions, squared = False) # retorna a raiz quadradica do erro-medio
print("RMSE:", rmse)
print("------------- Execução finalizada ---------------")


#### Segunda execução: Com informações dos artigos:
print("------------- Segunda execução: Com informações adicionais dos artigos: ---------------")
df_train, df_test = get_data_citeulike(df_ratings_complete)
model = GraphRec(df_train, df_test, ItemData=True, UserData = False, Graph=False, Dataset='citeU')
print("Execução da predição para teste")
df_test = df_test.sort_values('user') # retorna o dataset de treino ordenado com base nos ids
qids = df_test['user'] # qids sao os ids dos usuarios
y_test = df_test['rate'] # y_test sao os scores verdadeiros do teste
predictions = np.array(model.predict(df_test)).flatten() # model.predict(df_test) 
print("Primeiras 10 predições: " , predictions[:10])
print("Resultados Segunda Execução: ")

ndcgs = queries_ndcg(y_test, predictions, qids) # retorna uma lista com ndcg de cada query (que seria id de cada usuario)
print("MEAN NDCGS:", ndcgs.mean())
rmse = mean_squared_error(y_test, predictions, squared = False) # retorna a raiz quadradica do erro-medio
print("RMSE:", rmse)
print("------------- Execução finalizada ---------------")
