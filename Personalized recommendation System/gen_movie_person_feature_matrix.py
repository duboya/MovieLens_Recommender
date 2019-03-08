# -*- coding = utf-8 -*-
"""
Main function to generate the movie2movie matrix and person2person matrix.

@author: dby_freedom
"""
import os
import pickle

import numpy as np
import tensorflow as tf

from tools import load_config, load_params
from model_trainer import train_fn

ProcessedDataDir = './processed_data'

try:
    load_dir = load_params()
except FileNotFoundError:
    train_fn()
    load_dir = load_params()
title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(
    open(ProcessedDataDir + os.sep + 'preprocess.p', mode='rb'))

embed_dim, uid_max, gender_max, age_max, job_max, movie_id_max, movie_categories_max, \
movie_title_max, combiner, sentences_size, window_sizes, filter_num = load_config()

# 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}


def get_tensors(loaded_graph):
    uid = loaded_graph.get_tensor_by_name("uid:0")
    user_gender = loaded_graph.get_tensor_by_name("user_gender:0")
    user_age = loaded_graph.get_tensor_by_name("user_age:0")
    user_job = loaded_graph.get_tensor_by_name("user_job:0")
    movie_id = loaded_graph.get_tensor_by_name("movie_id:0")
    movie_categories = loaded_graph.get_tensor_by_name("movie_categories:0")
    movie_titles = loaded_graph.get_tensor_by_name("movie_titles:0")
    targets = loaded_graph.get_tensor_by_name("targets:0")
    dropout_keep_prob = loaded_graph.get_tensor_by_name("dropout_keep_prob:0")
    lr = loaded_graph.get_tensor_by_name("LearningRate:0")
    # 两种不同计算预测评分的方案使用不同的name获取tensor inference
    #     inference = loaded_graph.get_tensor_by_name("inference/inference/BiasAdd:0")
    inference = loaded_graph.get_tensor_by_name(
        "inference/ExpandDims:0")  # 之前是MatMul:0 因为inference代码修改了 这里也要修改 感谢网友 @清歌 指出问题
    movie_combine_layer_flat = loaded_graph.get_tensor_by_name("movie_fc/Reshape:0")
    user_combine_layer_flat = loaded_graph.get_tensor_by_name("user_fc/Reshape:0")
    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat


def gen_movie_matrix():
    loaded_graph = tf.Graph()  #
    movie_matrics = []
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # Get Tensors from loaded model
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, _, movie_combine_layer_flat, __ = get_tensors(
            loaded_graph)  # loaded_graph

        for item in movies.values:
            categories = np.zeros([1, 18])
            categories[0] = item.take(2)

            titles = np.zeros([1, sentences_size])
            titles[0] = item.take(1)

            feed = {
                movie_id: np.reshape(item.take(0), [1, 1]),
                movie_categories: categories,  # x.take(6,1)
                movie_titles: titles,  # x.take(5,1)
                dropout_keep_prob: 1}

            movie_combine_layer_flat_val = sess.run([movie_combine_layer_flat], feed)
            movie_matrics.append(movie_combine_layer_flat_val)

    pickle.dump((np.array(movie_matrics).reshape(-1, 200)), open(ProcessedDataDir + os.sep + 'movie_matrics.p', 'wb'))
    # movie_matrics = pickle.load(open(ProcessedDataDir + os.sep + 'movie_matrics.p', mode='rb'))
    return movie_matrics


def gen_user_matrix():
    loaded_graph = tf.Graph()  #
    users_matrics = []
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # Get Tensors from loaded model
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, _, __, user_combine_layer_flat = get_tensors(
            loaded_graph)  # loaded_graph

        for item in users.values:
            feed = {
                uid: np.reshape(item.take(0), [1, 1]),
                user_gender: np.reshape(item.take(1), [1, 1]),
                user_age: np.reshape(item.take(2), [1, 1]),
                user_job: np.reshape(item.take(3), [1, 1]),
                dropout_keep_prob: 1}

            user_combine_layer_flat_val = sess.run([user_combine_layer_flat], feed)
            users_matrics.append(user_combine_layer_flat_val)

    pickle.dump((np.array(users_matrics).reshape(-1, 200)), open(ProcessedDataDir + os.sep + 'users_matrics.p', 'wb'))
    # users_matrics = pickle.load(open(ProcessedDataDir + os.sep + 'users_matrics.p', mode='rb'))
    return users_matrics
