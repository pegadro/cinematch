from recommender.recommender import RatingsNormalizer
import numpy as np
from scipy import sparse

def get_ratings_binary_matrix(ratings_matrix):
    ratings_binary_matrix = np.where(ratings_matrix != 0, 1, ratings_matrix)
    
    return ratings_binary_matrix

def get_movies_ratings_means(ratings_matrix, ratings_binary_matrix):
    ratings_normalizer = RatingsNormalizer()
    ratings_normalizer.fit(ratings_matrix, R=ratings_binary_matrix)
    
    return ratings_normalizer.means_[:,np.newaxis]

def predict_full_ratings(movies_latent_factors, users_latent_factors, intercept_latent_factors):
    predictions = np.matmul(movies_latent_factors, np.transpose(users_latent_factors)) + intercept_latent_factors
    
    return predictions