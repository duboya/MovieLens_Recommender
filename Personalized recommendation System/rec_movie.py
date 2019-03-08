# -*- coding = utf-8 -*-
"""
Main function to recommend movies .

@author: dby_freedom
"""
import os
import pickle
import random

import numpy as np
import tensorflow as tf

from tools import load_config
from model_trainer import train_fn
from gen_movie_person_feature_matrix import gen_movie_matrix, gen_user_matrix
from preprocess_data import load_data

ProcessedDataDir = './processed_data'
load_dir = './save'

try:
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(
        open(ProcessedDataDir + os.sep + 'preprocess.p', mode='rb'))
except OSError:
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()

embed_dim, uid_max, gender_max, age_max, job_max, movie_id_max, movie_categories_max, \
movie_title_max, combiner, sentences_size, window_sizes, filter_num = load_config()

# 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}

try:
    movie_matrics = pickle.load(open(ProcessedDataDir + os.sep + 'movie_matrics.p', mode='rb'))
    users_matrics = pickle.load(open(ProcessedDataDir + os.sep + 'users_matrics.p', mode='rb'))
except OSError:
    movie_matrics = gen_movie_matrix()
    users_matrics = gen_user_matrix()


def recommend_same_type_movie(movie_id_val, top_k=30):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        try:
            loader = tf.train.import_meta_graph(load_dir + '.meta')
            loader.restore(sess, load_dir)
        except OSError:
            train_fn()
            loader = tf.train.import_meta_graph(load_dir + '.meta')
            loader.restore(sess, load_dir)
        norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keep_dims=True))
        normalized_movie_matrics = movie_matrics / norm_movie_matrics

        # 推荐同类型的电影
        probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
        sim = (probs_similarity.eval())
        #     results = (-sim[0]).argsort()[0:top_k]
        #     print(results)

        print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))
        print("以下是给您的推荐：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != 5:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
            print(val)
            print(movies_orig[val])

        return results


def recommend_your_favorite_movie(user_id_val, top_k=10):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        try:
            loader = tf.train.import_meta_graph(load_dir + '.meta')
            loader.restore(sess, load_dir)
        except OSError:
            train_fn()
            loader = tf.train.import_meta_graph(load_dir + '.meta')
            loader.restore(sess, load_dir)

        # 推荐您喜欢的电影
        probs_embeddings = (users_matrics[user_id_val - 1]).reshape([1, 200])

        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
        sim = (probs_similarity.eval())
        #     print(sim.shape)
        #     results = (-sim[0]).argsort()[0:top_k]
        #     print(results)

        #     sim_norm = probs_norm_similarity.eval()
        #     print((-sim_norm[0]).argsort()[0:top_k])

        print("以下是给您的推荐：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != 5:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
            print(val)
            print(movies_orig[val])

        return results


def recommend_other_favorite_movie(movie_id_val, top_k=20):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        try:
            loader = tf.train.import_meta_graph(load_dir + '.meta')
            loader.restore(sess, load_dir)
        except OSError:
            train_fn()
            loader = tf.train.import_meta_graph(load_dir + '.meta')
            loader.restore(sess, load_dir)

        probs_movie_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(users_matrics))
        favorite_user_id = np.argsort(probs_user_favorite_similarity.eval())[0][-top_k:]
        #     print(normalized_users_matrics.eval().shape)
        #     print(probs_user_favorite_similarity.eval()[0][favorite_user_id])
        #     print(favorite_user_id.shape)

        print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))

        print("喜欢看这个电影的人是：{}".format(users_orig[favorite_user_id - 1]))
        probs_users_embeddings = (users_matrics[favorite_user_id - 1]).reshape([-1, 200])
        probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movie_matrics))
        sim = (probs_similarity.eval())
        #     results = (-sim[0]).argsort()[0:top_k]
        #     print(results)

        #     print(sim.shape)
        #     print(np.argmax(sim, 1))
        p = np.argmax(sim, 1)
        print("喜欢看这个电影的人还喜欢看：")

        results = set()
        while len(results) != 5:
            c = p[random.randrange(top_k)]
            results.add(c)
        for val in (results):
            print(val)
            print(movies_orig[val])

        return results
