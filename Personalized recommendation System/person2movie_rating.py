# -*- coding = utf-8 -*-
"""
Main function to generate the person2movie matrix.

@author: dby_freedom
"""
import os
import pickle

import numpy as np
import tensorflow as tf

from preprocess_data import load_data
from tools import get_tensors, load_config
from model_trainer import train_fn

load_dir = './save'
ProcessedDataDir = './processed_data'

try:
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(
        open(ProcessedDataDir + os.sep + 'preprocess.p', mode='rb'))
except OSError:
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()

embed_dim, uid_max, gender_max, age_max, job_max, movie_id_max, movie_categories_max, \
movie_title_max, combiner, sentences_size, window_sizes, filter_num = load_config()

# 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}


def rating_movie(user_id_val, movie_id_val):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        try:
            # Load saved model
            loader = tf.train.import_meta_graph(load_dir + '.meta')
            loader.restore(sess, load_dir)
        except OSError:
            train_fn()
            loader = tf.train.import_meta_graph(load_dir + '.meta')
            loader.restore(sess, load_dir)
        # Get Tensors from loaded model
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference, _, __ = get_tensors(
            loaded_graph)  # loaded_graph

        categories = np.zeros([1, 18])
        categories[0] = movies.values[movieid2idx[movie_id_val]][2]

        titles = np.zeros([1, sentences_size])
        titles[0] = movies.values[movieid2idx[movie_id_val]][1]

        feed = {
            uid: np.reshape(users.values[user_id_val - 1][0], [1, 1]),
            user_gender: np.reshape(users.values[user_id_val - 1][1], [1, 1]),
            user_age: np.reshape(users.values[user_id_val - 1][2], [1, 1]),
            user_job: np.reshape(users.values[user_id_val - 1][3], [1, 1]),
            movie_id: np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
            movie_categories: categories,  # x.take(6,1)
            movie_titles: titles,  # x.take(5,1)
            dropout_keep_prob: 1}

        # Get Prediction
        inference_val = sess.run([inference], feed)

        return inference_val
