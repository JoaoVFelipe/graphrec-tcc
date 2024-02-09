import numpy as np
import pandas as pd
import random
from os.path import exists

from sklearn.metrics import mean_squared_error, average_precision_score
from utils.graphrec import GraphRec, get_data100k
from utils.metrics import queries_ndcg, mean_ap

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings('ignore')


POSITIVE_VALUE = 5
NEGATIVE_VALUE = 1

USER_NUM=1000
ITEM_NUM=6000

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

    test_p = len(df_test[df_test["rate"]==POSITIVE_VALUE])
    test_n = len(df_test[df_test["rate"]==NEGATIVE_VALUE])

    train_p = len(df_train[df_train["rate"]==POSITIVE_VALUE])
    train_n = len(df_train[df_train["rate"]==NEGATIVE_VALUE])

    print("ITENS POSITIVOS EM TESTE: ", test_p)
    print("ITENS POSITIVOS EM TREINO: ", train_p)

    print("ITENS NEGATIVOS EM TESTE: ", test_n)
    print("ITENS NEGATIVOS EM TREINO: ", train_n)

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

def get_data100k():
    df = read_process("data/ml100k/u.data", sep="\t")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test

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
    df_ratings_sample = reduce_dimensionality(df_ratings, USER_NUM, ITEM_NUM)
    del df_ratings

    docs = df_ratings_sample['item'].unique()
    users = df_ratings_sample['user'].unique()
    rows_list = []
    ##df_ratings_complete = df_ratings_complete.iloc[0:0]

    print("Usuário e documentos finais:", users.size, docs.size)
    count = 0

    print("Adicionando itens não explicitos a matriz...")
    count = 0
    ### Cria todas as conexões restantes não apresentadas no dataset inicial - Rate == 0
    NEGATIVE_RATIO = 50
    for ind in users:
        user_id = ind
        user_ratings = list(map(int, df_ratings_sample.loc[df_ratings_sample['user'] == user_id, 'item']))
        negative_docs = [doc for doc in docs if doc not in user_ratings]
        # print("Process nr. ", count," ### User -- ", ind, "with", len(user_ratings), "items")
        ### Adiciona todos os itens positivos a lista
        for positive_doc in user_ratings:  
            user_rating = [user_id, positive_doc, 1]
            # print("Positive docs", positive_doc)
            rows_list.append(user_rating)
            ### Para cada item positivo, pega uma quantidade negativa baseado no ratio 1:NEGATIVE RATIO
            random.seed(count)
            negative_list = random.sample(negative_docs, NEGATIVE_RATIO)
            # print("Negative docs", negative_list)
            for negative_doc in negative_list:
                negative_user_rating = [user_id, negative_doc, 0]
                rows_list.append(negative_user_rating)
            count = count+1

    ### Dataset completo
    print("Processamento completo. Criando dataframe com ", len(rows_list), " items.")
    ### Substitui os valores de 0 e 1 para valores mais compreensiveis pelo graphrec
    df_ratings_complete = pd.DataFrame(columns=['user', 'item', 'rate'], data=rows_list)
    df_ratings_complete.loc[df_ratings_complete['rate'] == 1, 'rate'] = POSITIVE_VALUE
    df_ratings_complete.loc[df_ratings_complete['rate'] == 0, 'rate'] = NEGATIVE_VALUE
    # random_sampling = df_ratings_complete.groupby("rate").sample(n=14000, random_state=42)
    print("Limpando dados...")
    del rows_list
    del docs
    del users
    del df_ratings_sample
    return df_ratings_complete


### Check if pre processed dataset is already saved
preprocessed_path = 'preprocess/pre_processed_dataset-1000:5000.csv'
file_exists = exists(preprocessed_path)

if(not file_exists):
    df_ratings_complete = pre_process_dataset()
    df_ratings_complete.to_csv(preprocessed_path)
else:
    df_ratings_complete = pd.read_csv(preprocessed_path)

#### Primeira execução sem informações dos artigos:
print("------------- Primeira execução: Sem informações adicionais dos artigos: ---------------")
df_train, df_test = get_data_citeulike(df_ratings_complete)
model = GraphRec(df_train, df_test, ItemData=False, UserData = False, Graph=True, Dataset='citeU', USER_NUM=USER_NUM, ITEM_NUM=ITEM_NUM) 
print("------------- Execução finalizada ---------------")
print("Execução da predição para teste")
df_test = df_test.sort_values('user') # retorna o dataset de treino ordenado com base nos ids
qids = df_test['user'] # qids sao os ids dos usuarios
y_test = df_test['rate'] # y_test sao os scores verdadeiros do teste
predictions = np.array(model.predict(df_test)).flatten() # model.predict(df_test) 

print("Resultados Primeira Execução: ")
ndcgs = queries_ndcg(y_test, predictions, qids) # retorna uma lista com ndcg de cada query (que seria id de cada usuario)
print("MEAN NDCGS:", ndcgs.mean())
rmse = mean_squared_error(y_test, predictions, squared = False) # retorna a raiz quadradica do erro-medio
print("RMSE:", rmse)
map_res = mean_ap(y_test, predictions, qids, 5.0)
print("MAP:", map_res)
# results_ndcg = pd.DataFrame(columns=['user', 'real_values', 'predicted_values'], data=result_list)
# results_ndcg.to_csv('results/results_noadd.csv')
print("------------- Execução finalizada ---------------")

#### Segunda execução: Com informações dos artigos:
print("------------- Segunda execução: Com informações adicionais dos artigos: ---------------")
df_train, df_test = get_data_citeulike(df_ratings_complete)
model = GraphRec(df_train, df_test, ItemData=True, UserData = False, Graph=True, Dataset='citeU', USER_NUM=USER_NUM, ITEM_NUM=ITEM_NUM)
print("Execução da predição para teste")
df_test = df_test.sort_values('user') # retorna o dataset de treino ordenado com base nos ids
qids = df_test['user'] # qids sao os ids dos usuarios
y_test = df_test['rate'] # y_test sao os scores verdadeiros do teste
predictions = np.array(model.predict(df_test)).flatten() # model.predict(df_test) 
print("Resultados Segunda Execução: ")
print("Primeiras 10 predições: " , predictions[:10])
print("Primeiros 10 items: " , y_test[:10])
ndcgs = queries_ndcg(y_test, predictions, qids) # retorna uma lista com ndcg de cada query (que seria id de cada usuario)
print("MEAN NDCGS:", ndcgs.mean())
rmse = mean_squared_error(y_test, predictions, squared = False) # retorna a raiz quadradica do erro-medio
print("RMSE:", rmse)
map_res = mean_ap(y_test, predictions, qids, 5.0)
print("MAP:", map_res)
# results_ndcg = pd.DataFrame(columns=['user', 'real_values', 'predicted_values'], data=result_list)
# results_ndcg.to_csv('results/results_addinfo.csv')
print("------------- Execução finalizada ---------------")
