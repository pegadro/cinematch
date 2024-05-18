import pandas as pd

def load_datasets():
    links = pd.read_csv("datasets/ml-latest-small/links.csv")
    movies = pd.read_csv("datasets/ml-latest-small/movies.csv")
    ratings = pd.read_csv("datasets/ml-latest-small/ratings.csv")
    tags = pd.read_csv("datasets/ml-latest-small/tags.csv")
    
    return links, movies, ratings, tags