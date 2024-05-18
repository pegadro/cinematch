import pandas as pd
import numpy as np

def load_links():
    return pd.read_csv("datasets/ml-latest-small/links.csv")

def load_movies():
    return pd.read_csv("datasets/ml-latest-small/movies.csv")
    
def load_ratings():
    return pd.read_csv("datasets/ml-latest-small/ratings.csv")

def load_tags():
    return pd.read_csv("datasets/ml-latest-small/tags.csv")

def get_n_users(ratings):
    return len(ratings["userId"].unique())

def get_n_movies(movies):
    return len(movies["movieId"])

def load_movies_enhanced():
    movies = load_movies()
    ratings = load_ratings()
    
    #ratings_counting = ratings.groupby("movieId").size().reset_index(name="ratings")
    #movies = movies.merge(ratings_counting, on="movieId", how="left")
    
    aggregates = ratings.groupby("movieId").agg(
        ratings=pd.NamedAgg(column="rating", aggfunc="size"),
        mean_rating=pd.NamedAgg(column="rating", aggfunc="mean")
    ).reset_index()
    
    movies = movies.merge(aggregates, on="movieId", how="left")
    
    movies["ratings"].fillna(0, inplace=True)
    movies["mean_rating"].fillna(0, inplace=True)
    
    return movies

def load_datasets():
    links = load_links()
    movies = load_movies()
    ratings = load_ratings()
    tags = load_tags()
    
    return links, movies, ratings, tags

def get_matrix_ratings(ratings, movies):
    n_users = get_n_users(ratings)
    n_movies = get_n_movies(movies)
    
    ratings_matrix = np.zeros((n_movies, n_users))

    for i_movie in range(n_movies):
        movie = ratings[ratings["movieId"] == i_movie + 1]
        
        movie_ratings = np.zeros((n_users,))

        for index, row in movie.iterrows():
            movie_ratings[int(row["userId"]-1)] = row["rating"]
            
        ratings_matrix[i_movie] = movie_ratings
        
    return ratings_matrix

def get_matrix_rated(matrix_ratings):
    return np.where(matrix_ratings != 0, 1, matrix_ratings)

def normalize_matrix_ratings(matrix_ratings, matrix_rated):
    
    means = []
    for i in range(matrix_rated.shape[0]):
        indexes = matrix_rated[i] == 1
        means.append(matrix_ratings[i][indexes].mean() if indexes.any() else 0)
    
    means = np.array(means)
    matrix_ratings_mean_normalized = matrix_ratings.copy()
    
    for i in range(matrix_ratings.shape[0]):
        for j in range(matrix_ratings.shape[1]):
            if matrix_rated[i,j] == 1:
                matrix_ratings_mean_normalized[i,j] -= means[i]
    
    return matrix_ratings_mean_normalized, means[:,np.newaxis]

def get_genre_list(movies):
    genres = []
    for movie in range(get_n_movies(movies)):
        genres += movies["genres"][movie].split('|')
        
    genres = np.unique(genres)
    return genres

