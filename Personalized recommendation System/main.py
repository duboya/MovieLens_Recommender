# -*- coding = utf-8 -*-
"""
Main function to personal recommendation systems.

@author: dby_freedom
"""
from person2movie_rating import rating_movie
from rec_movie import *

ProcessedData = './processed_data'
InputUser = '../data/ml-1m/users.dat'
InputMv = '../data/ml-1m/movies.dat'
InputRating = '../data/ml-1m/ratings.dat'


def main():
    # rating_mv = rating_movie(234, 1401)
    print(rating_movie(234, 1401))

    recommend_same_type_movie(1401, 20)
    recommend_your_favorite_movie(234, 10)
    recommend_other_favorite_movie(1401, 30)


if __name__ == "__main__":
    main()
