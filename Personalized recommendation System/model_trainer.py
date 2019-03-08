# -*- coding = utf-8 -*-
"""
Main function to build personal recommendation systems.

@author: dby_freedom
"""
import datetime
import os
import pickle
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocess_data import load_data
from tools import load_config, save_params

ProcessedData = './processed_data/preprocess.p'
ProcessedParams = './processed_data/params.p'
ProcessedDataDir = './processed_data'

try:
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(
        open(ProcessedDataDir + os.sep + 'preprocess.p', mode='rb'))
except OSError:
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()


embed_dim, uid_max, gender_max, age_max, job_max, movie_id_max, movie_categories_max, \
movie_title_max, combiner, sentences_size, window_sizes, filter_num = load_config()

movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}

'''
Hyper parameters

'''

# Number of Epochs
num_epochs = 1
# Batch Size
batch_size = 256

dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 20

save_dir = './save'

'''
Define input placeholder

'''


def get_inputs():
    uid = tf.placeholder(tf.int32, [None, 1], name="uid")
    user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
    user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
    user_job = tf.placeholder(tf.int32, [None, 1], name="user_job")

    movie_id = tf.placeholder(tf.int32, [None, 1], name="movie_id")
    movie_categories = tf.placeholder(tf.int32, [None, 18], name="movie_categories")
    movie_titles = tf.placeholder(tf.int32, [None, 15], name="movie_titles")
    targets = tf.placeholder(tf.int32, [None, 1], name="targets")

    LearningRate = tf.placeholder(tf.float32, name="LearningRate")
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, LearningRate, dropout_keep_prob


'''
Define user's embedding matrix

'''


def get_user_embedding(uid, user_gender, user_age, user_job):
    with tf.name_scope("user_embedding"):
        uid_embed_matrix = tf.Variable(tf.random_uniform([uid_max, embed_dim], -1, 1), name="uid_embed_matrix")
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name="uid_embed_layer")

        gender_embed_matrix = tf.Variable(tf.random_uniform([gender_max, embed_dim // 2], -1, 1),
                                          name="gender_embed_matrix")
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name="gender_embed_layer")

        age_embed_matrix = tf.Variable(tf.random_uniform([age_max, embed_dim // 2], -1, 1), name="age_embed_matrix")
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name="age_embed_layer")

        job_embed_matrix = tf.Variable(tf.random_uniform([job_max, embed_dim // 2], -1, 1), name="job_embed_matrix")
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name="job_embed_layer")
    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer


def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
    with tf.name_scope("user_fc"):
        # 第一层全连接
        uid_fc_layer = tf.layers.dense(uid_embed_layer, embed_dim, name="uid_fc_layer", activation=tf.nn.relu)
        gender_fc_layer = tf.layers.dense(gender_embed_layer, embed_dim, name="gender_fc_layer", activation=tf.nn.relu)
        age_fc_layer = tf.layers.dense(age_embed_layer, embed_dim, name="age_fc_layer", activation=tf.nn.relu)
        job_fc_layer = tf.layers.dense(job_embed_layer, embed_dim, name="job_fc_layer", activation=tf.nn.relu)

        # 第二层全连接
        user_combine_layer = tf.concat([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer], 2)  # (?, 1, 128)
        user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer, 200, tf.tanh)  # (?, 1, 200)

        user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])
    return user_combine_layer, user_combine_layer_flat


def get_movie_id_embed_layer(movie_id):
    with tf.name_scope("movie_embedding"):
        movie_id_embed_matrix = tf.Variable(tf.random_uniform([movie_id_max, embed_dim], -1, 1),
                                            name="movie_id_embed_matrix")
        movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name="movie_id_embed_layer")
    return movie_id_embed_layer


def get_movie_categories_layers(movie_categories):
    with tf.name_scope("movie_categories_layers"):
        movie_categories_embed_matrix = tf.Variable(tf.random_uniform([movie_categories_max, embed_dim], -1, 1),
                                                    name="movie_categories_embed_matrix")
        movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories,
                                                              name="movie_categories_embed_layer")
        if combiner == "sum":
            movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1, keep_dims=True)  # 进行行加和
    #     elif combiner == "mean":

    return movie_categories_embed_layer


'''
Movie Title的文本卷积网络实现

'''


def get_movie_cnn_layer(movie_titles, dropout_keep_prob):
    # 从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
    with tf.name_scope("movie_embedding"):
        movie_title_embed_matrix = tf.Variable(tf.random_uniform([movie_title_max, embed_dim], -1, 1),
                                               name="movie_title_embed_matrix")
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles,
                                                         name="movie_title_embed_layer")
        movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)

    # 对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
    pool_layer_lst = []
    for window_size in window_sizes:
        with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
            filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num], stddev=0.1),
                                         name="filter_weights")
            filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")

            conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand, filter_weights, [1, 1, 1, 1], padding="VALID",
                                      name="conv_layer")
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name="relu_layer")

            maxpool_layer = tf.nn.max_pool(relu_layer, [1, sentences_size - window_size + 1, 1, 1], [1, 1, 1, 1],
                                           padding="VALID", name="maxpool_layer")
            pool_layer_lst.append(maxpool_layer)

    # Dropout层
    with tf.name_scope("pool_dropout"):
        pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")
        max_num = len(window_sizes) * filter_num
        pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name="pool_layer_flat")

        dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name="dropout_layer")
    return pool_layer_flat, dropout_layer


'''
将Movie的各个层一起做全连接

'''


def get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
    with tf.name_scope("movie_fc"):
        # 第一层全连接
        movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, embed_dim, name="movie_id_fc_layer",
                                            activation=tf.nn.relu)
        movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, embed_dim,
                                                    name="movie_categories_fc_layer", activation=tf.nn.relu)

        # 第二层全连接
        movie_combine_layer = tf.concat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)  # (?, 1, 96)
        movie_combine_layer = tf.contrib.layers.fully_connected(movie_combine_layer, 200, tf.tanh)  # (?, 1, 200)

        movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])
    return movie_combine_layer, movie_combine_layer_flat


'''
构建计算图

'''


def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]


def train_fn():
    # 电影名长度，做词嵌入要求输入的维度是固定的，这里设置为 15
    # 长度不够用空白符填充，太长则进行截断
    sentences_size = title_count  # = 15

    # reset_default_graph 操作应该在 tensorflow 的其他所有操作之前进行，否则将会出现不可知的问题
    # tensorflow 中的 graph 包含的是一系列的操作和使用到这些操作的 tensor
    tf.reset_default_graph()
    train_graph = tf.Graph()
    # Graph 只是当前线程的属性，如果想要在其他的线程使用这个 Graph，那么就要像下面这样指定
    # 下面的 with 语句将 train_graph 设置为当前线程的默认 graph
    with train_graph.as_default():
        # 获取输入占位符
        uid, user_gender, user_age, user_job, \
        movie_id, movie_categories, movie_titles, \
        targets, lr, dropout_keep_prob = get_inputs()
        # 获取User的4个嵌入向量
        uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(uid, user_gender,
                                                                                                   user_age, user_job)
        # 得到用户特征
        user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer, gender_embed_layer,
                                                                             age_embed_layer, job_embed_layer)
        # 获取电影ID的嵌入向量
        movie_id_embed_layer = get_movie_id_embed_layer(movie_id)
        # 获取电影类型的嵌入向量
        movie_categories_embed_layer = get_movie_categories_layers(movie_categories)
        # 获取电影名的特征向量
        pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_titles, dropout_keep_prob)
        # 得到电影特征
        movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(movie_id_embed_layer,
                                                                                movie_categories_embed_layer,
                                                                                dropout_layer)
        # 计算出评分，要注意两个不同的方案，inference的名字（name值）是不一样的，后面做推荐时要根据name取得tensor
        # tensorflow 的 name_scope 指定了 tensor 范围，方便我们后面调用，通过指定 name_scope 来调用其中的 tensor
        with tf.name_scope("inference"):
            # 直接将用户特征矩阵和电影特征矩阵相乘得到得分，最后要做的就是对这个得分进行回归
            inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1)
            inference = tf.expand_dims(inference, axis=1)

        with tf.name_scope("loss"):
            # MSE损失，将计算值回归到评分
            cost = tf.losses.mean_squared_error(targets, inference)
            # 将每个维度的 cost 相加，计算它们的平均值
            loss = tf.reduce_mean(cost)
        # 优化损失
        #     train_op = tf.train.AdamOptimizer(lr).minimize(loss)  #cost
        # 在为 tensorflow 设置 name 参数的时候，是为了能在 graph 中看到什么变量进行了什么操作
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(lr)
        gradients = optimizer.compute_gradients(loss)  # cost
        train_op = optimizer.apply_gradients(gradients, global_step=global_step)

    losses = {'train': [], 'test': []}

    with tf.Session(graph=train_graph) as sess:

        # 搜集数据给tensorBoard用
        # Keep track of gradient values and sparsity
        grad_summaries = []
        for g, v in gradients:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
                # tf.nn.zero_fraction 用于计算矩阵中 0 所占的比重，也就是计算矩阵的稀疏程度
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':', '_')),
                                                     tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", loss)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Inference summaries
        inference_summary_op = tf.summary.merge([loss_summary])
        inference_summary_dir = os.path.join(out_dir, "summaries", "inference")
        inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch_i in range(num_epochs):

            # 将数据集分成训练集和测试集，随机种子不固定
            train_X, test_X, train_y, test_y = train_test_split(features,
                                                                targets_values,
                                                                test_size=0.2,
                                                                random_state=0)

            train_batches = get_batches(train_X, train_y, batch_size)
            test_batches = get_batches(test_X, test_y, batch_size)

            # 训练的迭代，保存训练损失
            for batch_i in range(len(train_X) // batch_size):
                x, y = next(train_batches)

                categories = np.zeros([batch_size, 18])
                for i in range(batch_size):
                    categories[i] = x.take(6, 1)[i]

                titles = np.zeros([batch_size, sentences_size])
                for i in range(batch_size):
                    titles[i] = x.take(5, 1)[i]

                feed = {
                    uid: np.reshape(x.take(0, 1), [batch_size, 1]),
                    user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
                    user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
                    user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
                    movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
                    movie_categories: categories,  # x.take(6,1)
                    movie_titles: titles,  # x.take(5,1)
                    targets: np.reshape(y, [batch_size, 1]),
                    dropout_keep_prob: dropout_keep,  # dropout_keep
                    lr: learning_rate}

                step, train_loss, summaries, _ = sess.run([global_step, loss, train_summary_op, train_op], feed)  # cost
                losses['train'].append(train_loss)
                train_summary_writer.add_summary(summaries, step)  #

                # Show every <show_every_n_batches> batches
                if batch_i % show_every_n_batches == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        time_str,
                        epoch_i,
                        batch_i,
                        (len(train_X) // batch_size),
                        train_loss))

            # 使用测试数据的迭代
            for batch_i in range(len(test_X) // batch_size):
                x, y = next(test_batches)

                categories = np.zeros([batch_size, 18])
                for i in range(batch_size):
                    categories[i] = x.take(6, 1)[i]

                titles = np.zeros([batch_size, sentences_size])
                for i in range(batch_size):
                    titles[i] = x.take(5, 1)[i]

                feed = {
                    uid: np.reshape(x.take(0, 1), [batch_size, 1]),
                    user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
                    user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
                    user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
                    movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
                    movie_categories: categories,  # x.take(6,1)
                    movie_titles: titles,  # x.take(5,1)
                    targets: np.reshape(y, [batch_size, 1]),
                    dropout_keep_prob: 1,
                    lr: learning_rate}

                step, test_loss, summaries = sess.run([global_step, loss, inference_summary_op], feed)  # cost

                # 保存测试损失
                losses['test'].append(test_loss)
                inference_summary_writer.add_summary(summaries, step)  #

                time_str = datetime.datetime.now().isoformat()
                if batch_i % show_every_n_batches == 0:
                    print('{}: Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
                        time_str,
                        epoch_i,
                        batch_i,
                        (len(test_X) // batch_size),
                        test_loss))
        save_params((save_dir))
        # Save Model
        saver.save(sess, save_dir)  # , global_step=epoch_i
        print('Model Trained and Saved')
