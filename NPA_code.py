#!/usr/bin/env python
# coding: utf-8

# In[8]:


from keras.optimizers import *
from keras import backend as K
from keras.models import Model
from keras.layers import *
import keras
import os
from numpy.linalg import cholesky
import pickle
import numpy as np
import itertools
import time
import datetime
from nltk.tokenize import word_tokenize  # word_tokenize 只是分词而已，并没有向量化！！！
import csv
import random
import nltk
nltk.download('punkt')


# In[9]:


def newsample(nnn, ratio):
    if ratio > len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1), ratio)  # 增幅数据再采样
    else:
        return random.sample(nnn, ratio)


# In[10]:


# userid    M/A     positive_ids #TAB# negative_ids #TAB# date_time #N# positive_ids #TAB# negative_ids #TAB# date_time     [test]positive...#N# negative...

def preprocess_user_file(file='ClickData_sample.tsv', npratio=4):  # npratio: K+1 中的 K=4
    userid_dict = {}
    with open(file) as f:
        userdata = f.readlines()
        print('user data len: {}'.format(len(userdata)))
    for user in userdata:
        line = user.strip().split('\t')
        userid = line[0]
        if userid not in userid_dict:
            userid_dict[userid] = len(userid_dict)

    all_train_id = []
    all_train_pn = []
    all_label = []

    all_test_id = []
    all_test_pn = []
    all_test_label = []
    all_test_index = []

    all_user_pos = []
    all_test_user_pos = []

    for user in userdata:
        line = user.strip().split('\t')
        userid = line[0]
        if len(line) == 4:

            impre = [x.split('#TAB#') for x in line[2].split('#N#')]
        if len(line) == 3:
            impre = [x.split('#TAB#') for x in line[2].split('#N#')]

        trainpos = [x[0].split() for x in impre]  # positive
        trainneg = [x[1].split() for x in impre]  # negtive

        # train_datetime = [x[2].split() for x in impre]

        poslist = list(itertools.chain(*(trainpos)))
        neglist = list(itertools.chain(*(trainneg)))

        # 那么对应关系呢

        if len(line) == 4:    # ok, 最后是测试数据
            testimpre = [x.split('#TAB#') for x in line[3].split('#N#')]
            testpos = [x[0].split() for x in testimpre]
            testneg = [x[1].split() for x in testimpre]

            for i in range(len(testpos)):
                sess_index = []
                sess_index.append(len(all_test_pn))
                posset = list(set(poslist))   # poslist没有更新，这句没必要放在for 里面。。
                allpos = [int(p) for p in random.sample(
                    posset, min(50, len(posset)))[:50]]  # 最多随机采样50个正样本
                allpos += [0]*(50-len(allpos))    # 后面补零

                for j in testpos[i]:
                    all_test_pn.append(int(j))
                    all_test_label.append(1)
                    all_test_id.append(userid_dict[userid])  # 反复append？？？
                    all_test_user_pos.append(allpos)

                for j in testneg[i]:
                    all_test_pn.append(int(j))
                    all_test_label.append(0)
                    all_test_id.append(userid_dict[userid])
                    all_test_user_pos.append(allpos)
                sess_index.append(len(all_test_pn))
                all_test_index.append(sess_index)

        for impre_id in range(len(trainpos)):
            for pos_sample in trainpos[impre_id]:

                pos_neg_sample = newsample(trainneg[impre_id], npratio)
                pos_neg_sample.append(pos_sample)
                temp_label = [0]*npratio+[1]
                temp_id = list(range(npratio+1))
                random.shuffle(temp_id)

                shuffle_sample = []
                shuffle_label = []
                for id in temp_id:
                    shuffle_sample.append(int(pos_neg_sample[id]))
                    shuffle_label.append(temp_label[id])

                posset = list(set(poslist)-set([pos_sample]))
                allpos = [int(p) for p in random.sample(
                    posset, min(50, len(posset)))[:50]]
                allpos += [0]*(50-len(allpos))
                all_train_pn.append(shuffle_sample)
                all_label.append(shuffle_label)
                all_train_id.append(userid_dict[userid])
                all_user_pos.append(allpos)

    all_train_pn = np.array(all_train_pn, dtype='int32')
    all_label = np.array(all_label, dtype='int32')
    all_train_id = np.array(all_train_id, dtype='int32')
    all_test_pn = np.array(all_test_pn, dtype='int32')
    all_test_label = np.array(all_test_label, dtype='int32')
    all_test_id = np.array(all_test_id, dtype='int32')
    all_user_pos = np.array(all_user_pos, dtype='int32')
    all_test_user_pos = np.array(all_test_user_pos, dtype='int32')
    return userid_dict, all_train_pn, all_label, all_train_id, all_test_pn, all_test_label, all_test_id, all_user_pos, all_test_user_pos, all_test_index


# In[11]:

#region xxx

def preprocess_news_file(file='DocMeta_sample.tsv'):
    with open(file) as f:
        newsdata = f.readlines()
        print('news data len: {}'.format(len(newsdata)))

    news = {}
    for newsline in newsdata:
        line = newsline.strip().split('\t')
        # map<id, [cate, category, tokenized_title]>
        news[line[1]] = [line[2], line[3], word_tokenize(line[6].lower())]
    word_dict_raw = {'PADDING': [0, 999999]}

    for docid in news:
        for word in news[docid][2]:
            if word in word_dict_raw:
                word_dict_raw[word][1] += 1
            else:
                # map<word, [current_map_size, word_count]>
                word_dict_raw[word] = [len(word_dict_raw), 1]
    word_dict = {}
    for i in word_dict_raw:
        if word_dict_raw[i][1] >= 2:  # 出现至少两次的单词
            # map<word, [current_map_size(index), word_count]> & word_count >= 2
            word_dict[i] = [len(word_dict), word_dict_raw[i][1]]
    print(len(word_dict), len(word_dict_raw))

    news_words = [[0]*30]
    news_index = {'0': 0}
    for newsid in news:
        word_id = []
        # map<news_id, index(0,1,2...)> ，写法好怪异！！！为什么不直接用enumerate()
        news_index[newsid] = len(news_index)
        for word in news[newsid][2]:
            if word in word_dict:
                word_id.append(word_dict[word][0])  # all indices
        word_id = word_id[:30]
        news_words.append(word_id+[0]*(30-len(word_id)))
    news_words = np.array(news_words, dtype='int32')
    # news_words & news_index all stored indices; news_words:dim3 ?
    return word_dict, news_words, news_index
    # 意思是标题总共30个单词?


# In[12]:


def get_embedding(word_dict):
    embedding_dict = {}
    cnt = 0
    # line0: word_id, line1,2,3...: word_embedding
    with open('/raid/glove.840B.300d.txt', 'rb')as f:
        linenb = 0
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            line = line.split()
            word = line[0].decode()
            linenb += 1
            if len(word) != 0:
                vec = [float(x) for x in line[1:]]
                if word in word_dict:
                    # 只对word_dict中出现过的词语保留embedding
                    embedding_dict[word] = vec
                    if cnt % 1000 == 0:
                        print(cnt, linenb, word)
                    cnt += 1

    # 这个初始化意义不明，哦懂了，就是可以直接用index给array赋值，不会出outofindex错误
    embedding_matrix = [0]*len(word_dict)
    cand = []  # 这是个什么玩意儿
    for i in embedding_dict:
        embedding_matrix[word_dict[i][0]] = np.array(
            embedding_dict[i], dtype='float32')   # index->word_embedding，就是这里的直接赋值
        # cand without word_dict's index
        cand.append(embedding_matrix[word_dict[i][0]])
    cand = np.array(cand, dtype='float32')  # cand is the matrix without order
    mu = np.mean(cand, axis=0)
    Sigma = np.cov(cand.T)
    # random multivariate distribution
    norm = np.random.multivariate_normal(mu, Sigma, 1)
    for i in range(len(embedding_matrix)):
        if type(embedding_matrix[i]) == int:  # 好家伙，用这个判断是不是空。。。
            embedding_matrix[i] = np.reshape(norm, 300)   # 如果是空就赋值正态分布数据
    embedding_matrix[0] = np.zeros(300, dtype='float32')   # 意义不明
    embedding_matrix = np.array(embedding_matrix, dtype='float32')
    print(embedding_matrix.shape)
    return embedding_matrix


# In[13]:


userid_dict, all_train_pn, all_label, all_train_id, all_test_pn, all_test_label, all_test_id, all_user_pos, all_test_user_pos, all_test_index = preprocess_user_file()

print('userid_dict: {}'.format(userid_dict))
print('all_train_pn: {}'.format(all_train_pn))

# In[]:
a = 20


# In[14]:


word_dict, news_words, news_index = preprocess_news_file()


# In[15]:


embedding_mat = get_embedding(word_dict)


# In[16]:

#endregion

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    # np.arange(x) == np.array(range(x))
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


# In[17]:


def generate_batch_data_train(all_train_pn, all_label, all_train_id, batch_size):
    inputid = np.arange(len(all_label))
    np.random.shuffle(inputid)
    y = all_label
    batches = [inputid[range(batch_size*i, min(len(y), batch_size*(i+1)))]
               for i in range(len(y)//batch_size+1)]

    while (True):
        for i in batches:
            # 这个split是什么鬼，i是一个数组
            # all_train_pn[i] 是和all_train_pn 同维数的
            # all_train_pn 是二维数组
            # all_train_pn 里是 K+1 训练小组的数组
            # 那 candidate 还tm是一个二维数组啊？？？？
            # news_words 是二维数组，wtf？？？？？？？？
            candidate = news_words[all_train_pn[i]]
            candidate_split = [candidate[:, k, :] for k in range(
                candidate.shape[1])]   # candidate[:,k,:] ????
            browsed_news = news_words[all_user_pos[i]]
            browsed_news_split = [browsed_news[:, k, :]
                                  for k in range(browsed_news.shape[1])]
            userid = np.expand_dims(all_train_id[i], axis=1)
            label = all_label[i]
            yield (candidate_split + browsed_news_split+[userid], label)
            # userid 用的是整数序号，evaluate 的时候，userid没见过，会不会影响embedding变得很奇怪？？？


# In[18]:


def generate_batch_data_test(all_test_pn, all_label, all_test_id, batch_size):
    inputid = np.arange(len(all_label))
    y = all_label
    batches = [inputid[range(batch_size*i, min(len(y), batch_size*(i+1)))]
               for i in range(len(y)//batch_size+1)]

    while (True):
        for i in batches:
            candidate = news_words[all_test_pn[i]]
            browsed_news = news_words[all_test_user_pos[i]]
            browsed_news_split = [browsed_news[:, k, :]
                                  for k in range(browsed_news.shape[1])]
            userid = np.expand_dims(all_test_id[i], axis=1)
            label = all_label[i]

            yield ([candidate] + browsed_news_split+[userid], label)


# In[19]:


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
npratio = 4
results = []


MAX_SENT_LENGTH = 30
MAX_SENTS = 50

user_id = Input(shape=(1,), dtype='int32')
user_embedding_layer = Embedding(len(userid_dict), 50, trainable=True)    # 变化的
user_embedding = user_embedding_layer(user_id)
user_embedding_word = Dense(200, activation='relu')(user_embedding)
user_embedding_word = Flatten()(user_embedding_word)
user_embedding_news = Dense(200, activation='relu')(user_embedding)
user_embedding_news = Flatten()(user_embedding_news)


news_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(word_dict), 300, weights=[
                            embedding_mat], trainable=True)
embedded_sequences = embedding_layer(news_input)
embedded_sequences = Dropout(0.2)(embedded_sequences)

cnnouput = Convolution1D(filters=400, kernel_size=3,  padding='same',
                         activation='relu', strides=1)(embedded_sequences)
cnnouput = Dropout(0.2)(cnnouput)


attention_a = Dot((2, 1))(
    [cnnouput, Dense(400, activation='tanh')(user_embedding_word)])
attention_weight = Activation('softmax')(attention_a)
news_rep = keras.layers.Dot((1, 1))([cnnouput, attention_weight])
newsEncoder = Model([news_input, user_id], news_rep)


all_news_input = [keras.Input((MAX_SENT_LENGTH,), dtype='int32')
                  for _ in range(MAX_SENTS)]
browsed_news_rep = [newsEncoder([news, user_id]) for news in all_news_input]
browsed_news_rep = concatenate([Lambda(lambda x: K.expand_dims(
    x, axis=1))(news) for news in browsed_news_rep], axis=1)


attention_news = keras.layers.Dot((2, 1))(
    [browsed_news_rep, Dense(400, activation='tanh')(user_embedding_news)])
attention_weight_news = Activation('softmax')(attention_news)
user_rep = keras.layers.Dot((1, 1))([browsed_news_rep, attention_weight_news])


candidates = [keras.Input((MAX_SENT_LENGTH,), dtype='int32')
              for _ in range(1+npratio)]
candidate_vecs = [newsEncoder([candidate, user_id])
                  for candidate in candidates]
logits = [keras.layers.dot([user_rep, candidate_vec], axes=-1)
          for candidate_vec in candidate_vecs]
logits = keras.layers.Activation(keras.activations.softmax)(
    keras.layers.concatenate(logits))

model = Model(candidates+all_news_input+[user_id], logits)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001), metrics=['acc'])


candidate_one = keras.Input((MAX_SENT_LENGTH,))
candidate_one_vec = newsEncoder([candidate_one, user_id])
score = keras.layers.Activation(keras.activations.sigmoid)(
    keras.layers.dot([user_rep, candidate_one_vec], axes=-1))
model_test = keras.Model([candidate_one]+all_news_input+[user_id], score)

for ep in range(3):
    traingen = generate_batch_data_train(
        all_train_pn, all_label, all_train_id, 100)
    model.fit_generator(traingen, epochs=1,
                        steps_per_epoch=len(all_train_id)//100)
    testgen = generate_batch_data_test(
        all_test_pn, all_test_label, all_test_id, 100)
    click_score = model_test.predict_generator(
        testgen, steps=len(all_test_id)//100, verbose=1)
    from sklearn.metrics import roc_auc_score
    all_auc = []
    all_mrr = []
    all_ndcg = []
    all_ndcg2 = []
    for m in all_test_index:
        if np.sum(all_test_label[m[0]:m[1]]) != 0 and m[1] < len(click_score):
            all_auc.append(roc_auc_score(
                all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0]))
            all_mrr.append(
                mrr_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0]))
            all_ndcg.append(ndcg_score(
                all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0], k=5))
            all_ndcg2.append(ndcg_score(
                all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0], k=10))
    results.append([np.mean(all_auc), np.mean(all_mrr),
                   np.mean(all_ndcg), np.mean(all_ndcg2)])
    print(np.mean(all_auc), np.mean(all_mrr),
          np.mean(all_ndcg), np.mean(all_ndcg2))
