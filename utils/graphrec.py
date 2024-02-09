'''
paper titles "Attribute-Aware Non-Linear Co-Embeddings of Graph Features" Accepted in RecSys 2019
This code was implemented using python 3.5 and TensorFlow  1.7
We would like to thank "Guocong Song" because we utilized parts of his code from "songgc/TF-recomm" in our implementation
'''



import numpy as np
import pandas as pd
import time
from collections import deque

import tensorflow as tf
from six import next
from sklearn import preprocessing
import sys
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix

class ShuffleIterator(object):

    def __init__(self, inputs, batch_size=10):
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size,))
        out = self.inputs[ids, :]
        return [out[:, i] for i in range(self.num_cols)]


class OneEpochIterator(ShuffleIterator):
    def __init__(self, inputs, batch_size=10):
        super(OneEpochIterator, self).__init__(inputs, batch_size=batch_size)
        if batch_size > 0:
            self.idx_group = np.array_split(np.arange(self.len), np.ceil(self.len / batch_size))
        else:
            self.idx_group = [np.arange(self.len)]
        self.group_id = 0

    def next(self):
        if self.group_id >= len(self.idx_group):
            self.group_id = 0
            raise StopIteration
        out = self.inputs[self.idx_group[self.group_id], :]
        self.group_id += 1
        return [out[:, i] for i in range(self.num_cols)]

def inferenceDense(phase,user_batch, item_batch,idx_user,idx_item, user_num, item_num,UReg=0.05,IReg=0.1):
    with tf.device(DEVICE): 
        user_batch = tf.nn.embedding_lookup(idx_user, user_batch, name="embedding_user")
        item_batch = tf.nn.embedding_lookup(idx_item, item_batch, name="embedding_item")
        
        ul1mf=tf.layers.dense(inputs=user_batch, units=MFSIZE,activation=tf.nn.crelu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        il1mf=tf.layers.dense(inputs=item_batch, units=MFSIZE,activation=tf.nn.crelu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        InferInputMF=tf.multiply(ul1mf, il1mf)

        infer=tf.reduce_sum(InferInputMF, 1, name="inference")

        regularizer = tf.add(UW*tf.nn.l2_loss(ul1mf), IW*tf.nn.l2_loss(il1mf), name="regularizer")

    return infer, regularizer

def inferenceSparse(phase,user_batch, item_batch,uflength,iflength, user_num, item_num,UReg=0.05,IReg=0.1):
    umfl1=tf.get_variable('umfl1', [uflength, MFSIZE],dtype=tf.float64, initializer=tf.random_normal_initializer(stddev=0.01))
    umfl1b=tf.get_variable('umfl1b', [1, MFSIZE],dtype=tf.float64, initializer=tf.constant_initializer(0))

    imfl1=tf.get_variable('imfl1', [iflength, MFSIZE],dtype=tf.float64, initializer=tf.random_normal_initializer(stddev=0.01))
    imfl1b=tf.get_variable('imfl1b', [1, MFSIZE],dtype=tf.float64, initializer=tf.constant_initializer(0))

    ul1mf=tf.add(tf.sparse_tensor_dense_matmul(user_batch,umfl1),umfl1b)
    ul1mf=tf.nn.crelu(ul1mf)
    il1mf=tf.add(tf.sparse_tensor_dense_matmul(item_batch,imfl1),imfl1b)
    il1mf=tf.nn.crelu(il1mf)

    InferInputMF=tf.multiply(ul1mf, il1mf)

    infer=tf.reduce_sum(InferInputMF, 1, name="inference")

    regularizer = tf.add(UW*tf.nn.l2_loss(ul1mf), IW*tf.nn.l2_loss(il1mf), name="regularizer")
    return infer, regularizer

def optimization(infer, regularizer, rate_batch, learning_rate=0.0005, reg=0.1):
    with tf.device(DEVICE):
        global_step = tf.train.get_global_step()
        assert global_step is not None
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        cost = tf.add(cost_l2, regularizer)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    return cost, train_op

def optimizationSparse(infer, regularizer, rate_batch, learning_rate=0.0005, reg=0.1):
    global_step = tf.train.get_global_step()
    assert global_step is not None
    cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
    cost = tf.add(cost_l2, regularizer)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    return cost, train_op

def clip(x):
    return np.clip(x, 1.0, 5.0)

USER_NUM = 1000
ITEM_NUM = 10000
  
def GraphRec(train, test,ItemData=False,UserData=False,Graph=False,Dataset='100k', USER_NUM=USER_NUM, ITEM_NUM=ITEM_NUM):

    AdjacencyUsers = np.zeros((USER_NUM,ITEM_NUM), dtype=np.float32) #np.asarray([[0 for x in range(ITEM_NUM)] for y in range(USER_NUM)],dtype=np.float16)
    DegreeUsers = np.zeros((USER_NUM,1), dtype=np.float32)# np.asarray([[0 for x in range(1)] for y in range(USER_NUM)],dtype=np.float16)
    
    AdjacencyItems = np.zeros((ITEM_NUM,USER_NUM), dtype=np.float32) #np.asarray([[0 for x in range(USER_NUM)] for y in range(ITEM_NUM)],dtype=np.float16)
    DegreeItems =  np.zeros((ITEM_NUM,1), dtype=np.float32) #np.asarray([[0 for x in range(1)] for y in range(ITEM_NUM)],dtype=np.float16)
    for index, row in train.iterrows():
      userid=int(row['user'])
      itemid=int(row['item'])
      AdjacencyUsers[userid][itemid]=row['rate']/5.0
      AdjacencyItems[itemid][userid]=row['rate']/5.0
      DegreeUsers[userid][0]+=1
      DegreeItems[itemid][0]+=1
    
    DUserMax=np.amax(DegreeUsers) 
    DItemMax=np.amax(DegreeItems)
    DegreeUsers=np.true_divide(DegreeUsers, DUserMax)
    DegreeItems=np.true_divide(DegreeItems, DItemMax)
    
    AdjacencyUsers=np.asarray(AdjacencyUsers,dtype=np.float32)
    AdjacencyItems=np.asarray(AdjacencyItems,dtype=np.float32)
    
    if(Graph):
        UserFeatures= np.concatenate((np.identity(USER_NUM,dtype=np.bool_), AdjacencyUsers,DegreeUsers), axis=1) 
        ItemFeatures= np.concatenate((np.identity(ITEM_NUM,dtype=np.bool_), AdjacencyItems,DegreeItems), axis=1) 
    else:
        UserFeatures=np.identity(USER_NUM,dtype=np.bool_)
        ItemFeatures=np.identity(ITEM_NUM,dtype=np.bool_)

    if(UserData):
      if(Dataset=='1m'):
        UsrDat=get_UserData1M()
      if(Dataset=='100k'):
        UsrDat=get_UserData100k()
      UserFeatures=np.concatenate((UserFeatures,UsrDat), axis=1) 

    if(ItemData):
      if(Dataset=='1m'):
        ItmDat=get_ItemData1M()
      if(Dataset=='100k'):
        ItmDat=get_ItemData100k()
      if(Dataset=='citeU'):
        dataset = pd.concat([train, test])
        ItmDat=get_ItemDataCiteULike(dataset)  
      ItemFeatures=np.concatenate((ItemFeatures,ItmDat), axis=1) 

    UserFeaturesLength=UserFeatures.shape[1]
    ItemFeaturesLength=ItemFeatures.shape[1]

    print(UserFeatures.shape)
    print(ItemFeatures.shape)

    samples_per_batch = len(train) // BATCH_SIZE

    iter_train = ShuffleIterator([train["user"],train["item"],train["rate"]],batch_size=BATCH_SIZE)
    iter_test = OneEpochIterator([test["user"],test["item"],test["rate"]],batch_size=10000)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float64, shape=[None])
    phase = tf.placeholder(tf.bool, name='phase')
    
    
    w_user = tf.constant(UserFeatures,name="userids", shape=[USER_NUM,UserFeatures.shape[1]],dtype=tf.float64)
    w_item = tf.constant(ItemFeatures,name="itemids", shape=[ITEM_NUM, ItemFeatures.shape[1]],dtype=tf.float64)


    infer, regularizer = inferenceDense(phase,user_batch, item_batch,w_user,w_item, user_num=USER_NUM, item_num=ITEM_NUM)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = optimization(infer, regularizer, rate_batch, learning_rate=LR, reg=0.09)

    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    finalerror=-1
    sess = tf.Session(config=config)
    # with tf.Session(config=config) as sess:
    if True:
        sess.run(init_op)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(EPOCH_MAX * samples_per_batch):
            #users, items, rates,y,m,d,dw,dy,w = next(iter_train)
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates,
                                                                   phase:True})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                degreelist=list()
                predlist=list()
                for users, items, rates in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items,                                                                                             
                                                            phase:False})

                    pred_batch = clip(pred_batch)            
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                finalerror=test_err
                print("{:3d},{:f},{:f},{:f}(s)".format(i // samples_per_batch, train_err, test_err, end - start))
                start = end
        return NNGraphrec(sess, user_batch, item_batch, rate_batch, phase, w_user, w_item, infer)

class NNGraphrec(object):

    def __init__(self, sess, user_batch, item_batch, rate_batch, phase, w_user, w_item, infer):
        self.sess = sess
        self.user_batch = user_batch
        self.item_batch = item_batch
        self.rate_batch = rate_batch
        self.phase = phase

        self.w_user = w_user
        self.w_item = w_item
        self.infer = infer

    def predict(self, test):
        iter_test = OneEpochIterator([test["user"], test["item"], test["rate"]], batch_size=1)
        all_predictions = []
        for users, items, rates in iter_test:
            pred_batch = self.sess.run(self.infer, feed_dict={self.user_batch: users,
                                                              self.item_batch: items,
                                                              self.phase: False})
            all_predictions.append(pred_batch)
        return all_predictions


def get_UserData1M():
    col_names = ["user", "gender", "age", "occupation","PostCode"]
    df = pd.read_csv('ml1m/users.dat', sep='::', header=None, names=col_names, engine='python')
    del df["PostCode"]
    df["user"]-=1
    df=pd.get_dummies(df,columns=[ "age", "gender", "occupation"])
    del df["user"]
    return df.values
  
def get_ItemData1M():
    ItemGenreMatrix= [[0 for x in range(18)] for y in range(ITEM_NUM)]
    ItemYearMatrix= [[0 for x in range(1)] for y in range(ITEM_NUM)]
    col_names = ["movieid", "movietitle","year", "Genre"]
    df = pd.read_csv('ml1m/newmovies.dat', sep='::', header=None, names=col_names, engine='python')
    del df["movietitle"]
    df["movieid"]-=1
    df=pd.concat([df,df['Genre'].str.get_dummies(sep='|').add_prefix('Genre_').astype('int8')],axis=1) 
    del  df["Genre"]
    df['year']-=df['year'].min()
    df['year']/=df['year'].max()
    df['year']=df['year'].fillna(0.0)    
    
    movieIdList=df["movieid"].values 
    movieGenres=df.drop(['movieid','year'],axis=1 ).values 
    movieYears=df['year'].values
    for indx in range(len(movieIdList)):
      mvid=movieIdList[indx]
      mvgenre=movieGenres[indx]
      mvyear=movieYears[indx]
      ItemGenreMatrix[mvid]=mvgenre
      ItemYearMatrix[mvid]=[mvyear]
    
    ItemGenreMatrix=np.asarray(ItemGenreMatrix)
    ItemYearMatrix=np.asarray(ItemYearMatrix)
    ItemFeatures=np.concatenate((ItemGenreMatrix, ItemYearMatrix), axis=1)
    return ItemGenreMatrix 

def get_UserData100k():
    col_names = ["user", "age", "gender", "occupation","PostCode"]
    df = pd.read_csv('ml100k/u.user', sep='|', header=None, names=col_names, engine='python')
    del df["PostCode"]
    df["user"]-=1
    df=pd.get_dummies(df,columns=[ "age", "gender", "occupation"])
    del df["user"]
    return df.values

def get_ItemData100k():
    col_names = ["movieid", "movietitle", "releasedate", "videoreleasedate","IMDbURL"
                ,"unknown","Action","Adventure","Animation","Childrens","Comedy","Crime","Documentary"
                ,"Drama","Fantasy","FilmNoir","Horror","Musical","Mystery","Romance","SciFi","Thriller"
                ,"War","Western"]
    df = pd.read_csv('ml100k/u.item', sep='|', header=None, names=col_names, engine='python')
    df['releasedate'] = pd.to_datetime(df['releasedate'])
    df['year'],df['month']=zip(*df['releasedate'].map(lambda x: [x.year,x.month]))
    df['year']-=df['year'].min()
    df['year']/=df['year'].max()
    df['year']=df['year'].fillna(0.0)

    del df["month"]
    del df["movietitle"]
    del df["releasedate"]
    del df["videoreleasedate"]
    del df["IMDbURL"]
  
    df["movieid"]-=1
    del  df["movieid"]
    return df.values 
  
def read_process(filname, sep="\t"):
    col_names = ["user", "item", "rate", "st"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    df["user"] -= 1
    df["item"] -= 1
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    return df


def read_processaiv(filname, sep="\t"):
    col_names = ["user", "item", "rate"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    return df  

def get_data1M():
    global PERC
    df = read_process("ml1m/ratings.dat", sep="::")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * PERC)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test

def get_dataaiv():
    global PERC
    df = read_process("aiv/ratings.dat", sep="::")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * PERC)
    df_train = df[0:split_index]
    df_testandval = df[split_index:].reset_index(drop=True)
    
    testandvalrows = len(df_testandval)
    split_indextest = int(testandvalrows * 0.5)
    df_val = df_testandval[0:split_indextest]
    df_test = df_testandval[split_indextest:].reset_index(drop=True)

    return df_train, df_test,df_val
  
def get_data100k():
    global PERC
    df = read_process("ml100k/u.data", sep="\t")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * PERC)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test


def get_citeulike_raw():
    colnames=['doc.id', 'title', 'citeulike.id', 'raw.title', 'raw.abstract'] 
    return pd.read_csv('data/citeulike/raw-data.csv', header=None, encoding='ISO-8859-1',  skiprows = 1, names=colnames)
     
def get_citations():
    return pd.read_csv('data/citeulike/citations.dat', header=None)

def get_item_tag():
     return pd.read_csv('data/citeulike/item-tag.dat', header=None)

def get_tags():
    return pd.read_csv('data/citeulike/tags.dat', header=None)

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
                df_reduced[tag_name] = False
            ### Ao invés de alterar no index, alterar no doc id
            df_reduced.at[index, tag_name] = True
        del item_tags

    df_reduced.drop('doc.id', axis=1)
    print("Processing complete")
    print("Cleaning data...")
    del df_item_tags
    del tags
    print("Sorting and filling indexes...")
    df_reduced = df_reduced.sort_index()
    df_reduced = df_reduced.reindex(range(df_reduced.index[0], df_reduced.index[-1] + 1), fill_value=False)
    return df_reduced.values

'''
HYPERPARAMETROS do modelo
'''

DEVICE="/cpu:0"

############# ML 100k dataset ###########

# BATCH_SIZE = 1000
# PERC=0.9
# USER_NUM = 943
# ITEM_NUM = 1682
# df_train, df_test = get_data100k()

# #Without Graph Feature
# MFSIZE=40
# UW=0.08
# IW=0.06
# LR=0.0002
# EPOCH_MAX = 601
# tf.reset_default_graph()
# GraphRec(df_train, df_test,ItemData=False,UserData=False,Graph=False,Dataset='100k')

# #With Graph Features
# MFSIZE=50
# UW=0.05
# IW=0.02
# LR=0.00003
# EPOCH_MAX = 196
# tf.reset_default_graph()
# GraphRec(df_train, df_test,ItemData=False,UserData=False,Graph=True,Dataset='100k')

#############################################

# CiteULike dataset

BATCH_SIZE = 1000
PERC = 0.9
# df_train, df_test = get_data100k()

#With Graph Feature w external features
MFSIZE=40
UW=0.08
IW=0.06
LR=0.0002
EPOCH_MAX = 200