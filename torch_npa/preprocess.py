# use dataset MINDSMALL

import numpy as np
import random
from itertools import chain
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from pathlib import Path
import logging
import pandas as pd
import nltk
import pickle
import math
from consts import MAX_NEWS_TITLE_LENGTH, MAX_HISTORY_NEWS_LENGTH, K, FILE_BEHAVIORS_TRAIN, FILE_BEHAVIORS_VAL

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

logger = logging.getLogger('torch_npa')

# read files
PATH_SMALL_TRAIN = Path('/raid/mindsmall_train')
PATH_SMALL_VAL = Path('/raid/mindsmall_dev')
PATH_GLOVE = Path('/raid/glove.840B.300d.txt')
FILE_BEHAVIORS = 'behaviors.tsv'
FILE_NEWS = 'news.tsv'
COLUMNS_BEHAVIORS = ['id', 'user_id', 'date_time', 'history', 'behavior']
COLUMNS_NEWS = ['news_id', 'category', 'sub_category', 'title',
                'abstract', 'url', 'title_entities', 'abstract_entities']

FILE_NEWS_WORDS = 'news_words.npy'
FILE_WORD_DICT = 'word_dict.dict'
FILE_EMBEDDING_MAT = 'embedding_mat.npy'
FILE_NEWS_INDEX = 'news_index.dict'


user_dict = {}
user_length = 0


def process_behavior(x, label=1):
    return [
        z[0]
        for z in
        [
            y.split('-')
            for y in str(x).split(' ')
        ]
        if int(z[1]) == label
    ]   # magic code


# 1
def read_data_from_files():
    behaviors_train = pd.read_csv(
        PATH_SMALL_TRAIN / FILE_BEHAVIORS, sep='\t', index_col=1, names=COLUMNS_BEHAVIORS)
    news_train = pd.read_csv(PATH_SMALL_TRAIN / FILE_NEWS,
                             sep='\t', index_col=0, names=COLUMNS_NEWS)
    news_train.dropna()
    # logger.info('be history: {}'.format(behaviors_train['history']))

    behaviors_val = pd.read_csv(
        PATH_SMALL_VAL / FILE_BEHAVIORS, sep='\t', index_col=1, names=COLUMNS_BEHAVIORS)
    behaviors_val.dropna()
    behaviors_val = behaviors_val.iloc[:int(len(behaviors_val) * 0.1)]
    news_val = pd.read_csv(PATH_SMALL_VAL / FILE_NEWS,
                           sep='\t', index_col=0, names=COLUMNS_NEWS)
    news_val.dropna()

    news_all = pd.concat([news_train, news_val])

    logger.info('news_train shape: {}, news_val shape: {}, news_all shape: {}'.format(
        news_train.shape,
        news_val.shape,
        news_all.shape))
    return behaviors_train, news_train, behaviors_val, news_val, news_all

# glove_pretrained = pd.read_csv()


def newsample(nnn, ratio):
    if ratio > len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1), ratio)  # 增幅数据再采样
    else:
        return random.sample(nnn, ratio)


def process_behaviors_data(behaviors: pd.DataFrame, mode='train'):
    # ？？？？？？？为什么tmd 这个user_length 需要这样？？？？其他的怎么没这回事？？？？？
    global user_length
    logger.info('processing behaviors')

    # remove those who has no history...
    behaviors['history'].replace('', np.nan, inplace=True)
    behaviors.dropna(subset=['history'], inplace=True)

    behaviors['history'] = behaviors['history'].apply(
        lambda x: str(x).split(' '))
    behaviors['pos_behavior'] = behaviors['behavior'].apply(
        process_behavior, args=(1,))
    behaviors['neg_behavior'] = behaviors['behavior'].apply(
        process_behavior, args=(0,))
    data = []
    for row_index, b in tqdm(behaviors.iterrows(), total=len(behaviors)):
        # user_dict
        if row_index not in user_dict:
            user_dict[row_index] = user_length
            # 因为去写它了，所以需要global！！！但是报错的位置是上一句？？？！！？！？！？fuck
            user_length += 1

        tmp = []
        if mode == 'train':
            # K+1 candidates
            for i in b['pos_behavior']:
                tmp2 = list(chain([i], newsample(b['neg_behavior'], K)))
                indices = list(range(K+1))
                random.shuffle(indices)
                candidates = []
                labels = []
                for x in indices:
                    candidates.append(tmp2[x])
                    if x == 0:
                        labels.append(1)
                    else:
                        labels.append(0)

                tmp.append(
                    {'user_id': user_dict[row_index], 'history': b['history'], 'candidate': candidates, 'label': labels})

        elif mode == 'val':
            # for i in b['pos_behavior']:
            #     # candidates.append(i)
            #     # labels.append(1)
            #     # print(b)
            #     tmp.append(
            #         {'user_id': user_dict[row_index], 'history': b['history'], 'candidate': i, 'label': 1})
            # for i in b['neg_behavior']:
            #     # candidates.append(i)
            #     # labels.append(0)
            #     tmp.append(
            #         {'user_id': user_dict[row_index], 'history': b['history'], 'candidate': i, 'label': 0})
            # # tmp.append(
            # #     {'user_id': user_dict[row_index], 'history': b['history'],
            # #         'candidate': candidates, 'label': labels}
            # # )


            candidates = []
            labels = []
            for i in b['pos_behavior']:
                candidates.append(i)
                labels.append(1)
            for i in b['neg_behavior']:
                candidates.append(i)
                labels.append(0)
            tmp.append(
                {'user_id': user_dict[row_index], 'history': b['history'],
                    'candidate': candidates, 'label': labels}
            )
        data.append(tmp)
    logger.info('concatenating...')
    data = pd.DataFrame(chain(*data))
    logger.debug('data: {}'.format(data.head(10)))
    return data


def process_news_data(news: pd.DataFrame):
    data = {}

    # get word embedding for titles, return map<news_id, title_embeddings>

    logger.info('processing news')
    word_dict_raw = {'PADDING': [0, 999999]}

    for news_id, n in tqdm(news.iterrows(), total=len(news)):
        title = n['title']
        # tokenize
        words = word_tokenize(title)
        for word in words:
            if word in word_dict_raw:
                word_dict_raw[word][1] += 1
            else:
                # 这个带上len()的操作很迷惑，表示该词在字典中的序号?干脆直接ordered_dict不就ok?
                word_dict_raw[word] = [len(word_dict_raw), 1]

        data[news_id] = words

    word_dict = {}
    for word in word_dict_raw:
        # 至少出现2次的单词
        if word_dict_raw[word][1] >= 2:
            word_dict[word] = [len(word_dict), word_dict_raw[word][1]]

    news_words = [[0] * MAX_NEWS_TITLE_LENGTH]
    news_index = {'0': 0}

    for id in tqdm(data):
        word_id = []
        news_index[id] = len(news_index)    # 随处都是这种写法，难受
        for word in data[id]:
            if word in word_dict:
                word_id.append(word_dict[word][0])

        word_id = word_id[:MAX_NEWS_TITLE_LENGTH]
        news_words.append(
            word_id + [0] * (MAX_NEWS_TITLE_LENGTH - len(word_id)))

    news_words = np.array(news_words, dtype='int32')

    # word_dict: 词对应次数
    # news_words: 每个新闻中每个词在word_dict中的位置？
    # 每个词的编号，可以和embedding matrix经过embedding层得到对应的embedding
    return word_dict, news_words, news_index
    # news_index 事实上后面没有用到，但是它存了新闻ID对应新闻编号的重要信息！


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

# 2


def get_train_val_data(behaviors_train, behaviors_val, news_all):
    behaviors_train_data = process_behaviors_data(
        behaviors_train, mode='train')
    behaviors_val_data = process_behaviors_data(behaviors_val, mode='val')

    if not Path(FILE_WORD_DICT).exists():
        word_dict, news_words, news_index = process_news_data(news_all)
        logger.debug('news_words: {}'.format(news_words))
        logger.debug('news_index: {}'.format(news_index))
        logger.debug('word_dict: {}'.format(word_dict))
        embedding_mat = get_embedding(word_dict)

        # save them somewhere...
        np.save(FILE_NEWS_WORDS, news_words)
        np.save(FILE_EMBEDDING_MAT, embedding_mat)
        with open(FILE_WORD_DICT, 'wb') as f:
            pickle.dump(word_dict, f)

        with open(FILE_NEWS_INDEX, 'wb') as f:
            pickle.dump(news_index, f)

        logger.info('saved news_words and word_dict')
    else:
        # load them
        news_words = np.load(FILE_NEWS_WORDS)
        with open(FILE_WORD_DICT, 'rb') as f:
            word_dict = pickle.load(f)

        with open(FILE_NEWS_INDEX, 'rb') as f:
            news_index = pickle.load(f)

        embedding_mat = np.load(FILE_EMBEDDING_MAT)

        logger.debug('news_words: {}'.format(news_words))

        logger.info('loaded word_dict and news_words')

    logger.debug('behaviors_train_data: \t{}'.format(
        behaviors_train_data.head()))
    logger.debug('behaviors_val_data: \t{}'.format(behaviors_val_data.head()))
    return behaviors_train_data, behaviors_val_data, word_dict, news_words, news_index, embedding_mat


# need to change news_id in behaviors to news_word representation
def replace_news_embedding_in_behaviors_lambda(history_item: list, news_words, news_index, mode='train', is_candidate=False):
    # if mode == 'train':
    #     # if type(history_item) is not list:D
    #     #     logger.error('??????? {} is not list ????'.format(history_item))
    #     #     return

    #     # for x in history_item:
    #     #     if type(x) is not str:
    #     #         logger.error(
    #     #             '???????? {} is not str ??????? in history_item: {}'.format(x, history_item))
    #     #     elif x == 'nan':
    #     #         # logger.error('??? nan 是个什么鬼 ???, {}'.format(history_item))
    #     #         return

    #     history = [
    #         # news_words, news_index 用的风生水起？？为什么不需要global？？？？
    #         news_words[news_index[news_id]]
    #         for news_id in history_item
    #         if news_id in news_index
    #     ]
    # else:
    #     history = news_words[news_index[history_item]]
    #     return history

    # if not is_candidate:
    #     history = history[:MAX_HISTORY_NEWS_LENGTH]
    #     history += [[0] * MAX_NEWS_TITLE_LENGTH] * \
    #         (MAX_HISTORY_NEWS_LENGTH - len(history))

    # magic code:
    # train | candidate | return_list | padding
    # ------|-----------|-------------|---------
    #   Y   |     Y     |      Y      |    N
    #   Y   |     N     |      Y      |    Y
    #   N   |     Y     |  N(single)  |    N
    #   N   |     N     |      Y      |    Y
    # if (not is_candidate) or mode == 'train':
    #     history = [
    #         # news_words, news_index 用的风生水起？？为什么不需要global？？？？
    #         news_words[news_index[news_id]]
    #         for news_id in history_item
    #         if news_id in news_index
    #     ]

    # if is_candidate and mode == 'val':
    #     history = news_words[news_index[history_item]]

    # if not is_candidate:
    #     history = history[:MAX_HISTORY_NEWS_LENGTH]
    #     history += [[0] * MAX_NEWS_TITLE_LENGTH] * \
    #         (MAX_HISTORY_NEWS_LENGTH - len(history))

    # should always return list
    history = [
        # news_words, news_index 用的风生水起？？为什么不需要global？？？？
        news_words[news_index[news_id]]
        for news_id in history_item
        if news_id in news_index
    ]

    # not candidate, add padding
    if not is_candidate:
        history = history[:MAX_HISTORY_NEWS_LENGTH]
        history += [[0] * MAX_NEWS_TITLE_LENGTH] * \
            (MAX_HISTORY_NEWS_LENGTH - len(history))

    return history


def replace_news_embedding_in_behaviors(behaviors: pd.DataFrame, news_words, news_index, mode='train'):
    behaviors['history'] = behaviors['history'].apply(
        replace_news_embedding_in_behaviors_lambda, args=(news_words, news_index, mode, False))
    behaviors['candidate'] = behaviors['candidate'].apply(
        replace_news_embedding_in_behaviors_lambda, args=(news_words, news_index, mode, True))

    return behaviors

# logger.error(behaviors_train_data[behaviors_train_data.history.str == '[\'nan\']'])

# # FUCK!
# behaviors_train_data.to_csv('fuck.csv')
# behaviors_val_data.to_csv('fuck2.csv')

# 3


def get_train_data(behaviors_train_data, behaviors_val_data, news_words, news_index):
    behaviors_train_data = replace_news_embedding_in_behaviors(
        behaviors_train_data, news_words, news_index, mode='train')
    behaviors_val_data = replace_news_embedding_in_behaviors(
        behaviors_val_data, news_words, news_index, mode='val')

    logger.info('after replace news_id to word_vec:')

    logger.info('behaviors_train_data: \t{}'.format(
        behaviors_train_data.head()))
    logger.info('behaviors_val_data: \t{}'.format(behaviors_val_data.head()))
    return behaviors_train_data, behaviors_val_data

# 还是不要save 的好，直接方法里返回就行。。。save 和load 的时间多出跑一遍太多了
# save them !!!! >3G 的文件，我勒个大草
# behaviors_train_data.to_csv(FILE_BEHAVIORS_TRAIN)
# 23G !!!???
# behaviors_val_data.to_csv(FILE_BEHAVIORS_VAL)


def load_data():
    behaviors_train, news_train, behaviors_val, news_val, news_all = read_data_from_files()
    behaviors_train_data, behaviors_val_data, word_dict, news_words, news_index, embedding_mat = get_train_val_data(
        behaviors_train, behaviors_val, news_all)
    behaviors_train_data, behaviors_val_data = get_train_data(
        behaviors_train_data, behaviors_val_data, news_words, news_index)
    logger.debug('behaviors_train_data to numpy? {}'.format(
        behaviors_train_data.to_numpy()[0]))
    # 我tm直接返回numpy格式！
    return behaviors_train_data.to_numpy(), behaviors_val_data.to_numpy(), user_length, len(word_dict), embedding_mat


if __name__ == '__main__':
    load_data()
