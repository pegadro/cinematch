# Cinematch: a collaborative filtering movie recommender system

Author: Pedro Alvarado García

## Introduction
Cinematch is a collaborative filtering movie recommender system that implements a regression and classification model.
- Regression model: Predicts a continuos rating (0.5 - 5) of a certain user to a certain movie.
- Classification model: Predicts a category (like or not like) of certain user to a certain movie.

Collaborative filtering:
- The idea is to make recommendations based on the rating of users who gave similar ratings to you.
- Only user ratings for items are needed.

## Dataset: [MovieLens](https://grouplens.org/datasets/movielens/) Latests Dataset
- Approximately 33,000,000 ratings and 2,000,000 tag applications applied to 86,000 movies by 330,975 users.
- Our recommendation system is limited to 10,000 movies and 5,000 users.

## Preprocessing
The objective of preprocessing is to obtain a ratings matrix where each row represents a movie and each column a user. Each element of the matrix is ​​a user's rating of a movie.

[Preprocessing notebook](create_ratings_matrix.ipynb)

## Model
To train our model we only need the ratings matrix.

Also, we need to create our own custom training loop using the automatic differentiation tools from tensorflow.

You can check the code from the regression and classification model in [recommender.py](recommender/recommender.py).

## Model training
You can check the model training code for regression and classification in the next notebooks:
- regression: [regression/model_training.ipynb](regression/model_training.ipynb)
- classification: [classification/model_training.ipynb](classification/model_training.ipynb)

## Additional features
There are some notebooks whose purpose is to index data to mongodb. These notebooks are the ones that start with insert in its name.

Also, there is a `api` directory where is implemented an api using FastAPI whose purpose is to connect to mongodb, read the parameters model and compute the predictions and recommendations.

A web UI was built to see the recommendations and it was connected to the api (this web UI is not shown here).
