import numpy as np
from scipy import sparse


def create_bl_ratings_matrix(ratings_matrix, threshold=3.5):
    ratings_matrix = np.where(ratings_matrix == 0, -1, ratings_matrix)

    one_mask = (ratings_matrix >= threshold) & (ratings_matrix != -1)
    zero_mask = (ratings_matrix < threshold) & (ratings_matrix != -1)

    ratings_matrix[one_mask] = 1
    ratings_matrix[zero_mask] = 0

    return ratings_matrix


def create_binary_bl_ratings_matrix(bl_ratings_matrix):
    binary_bl_ratings_matrix = np.zeros(bl_ratings_matrix.shape)
    mask = (bl_ratings_matrix == 1) | (bl_ratings_matrix == 0)
    binary_bl_ratings_matrix[mask] = 1

    return binary_bl_ratings_matrix
