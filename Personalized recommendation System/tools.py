import os
import pickle

ProcessedParams = './processed_data/params.p'
ProcessedDataDir = './processed_data'


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open(ProcessedParams, 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open(ProcessedParams, mode='rb'))


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


def load_config():
    """
    Load NN config parameters
    :return: NN config parameters
    """
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(
        open('./processed_data' + os.sep + 'preprocess.p', mode='rb'))
    # 嵌入矩阵的维度
    embed_dim = 32
    # 用户ID个数
    uid_max = max(features.take(0, 1)) + 1  # 6040
    # 性别个数
    gender_max = max(features.take(2, 1)) + 1  # 1 + 1 = 2
    # 年龄类别个数
    age_max = max(features.take(3, 1)) + 1  # 6 + 1 = 7
    # 职业个数movieid2idx
    job_max = max(features.take(4, 1)) + 1  # 20 + 1 = 21

    # 电影ID个数
    movie_id_max = max(features.take(1, 1)) + 1  # 3952
    # 电影类型个数
    movie_categories_max = max(genres2int.values()) + 1  # 18 + 1 = 19
    # 电影名单词个数
    movie_title_max = len(title_set)  # 5216

    # 对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
    combiner = "sum"

    # 电影名长度
    sentences_size = title_count  # = 15
    # 文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
    window_sizes = {2, 3, 4, 5}
    # 文本卷积核数量
    filter_num = 8

    return embed_dim, uid_max, gender_max, age_max, job_max, movie_id_max, movie_categories_max, \
           movie_title_max, combiner, sentences_size, window_sizes, filter_num
