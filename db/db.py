import numpy as np
from scipy import sparse
from pymongo import MongoClient


def connect_to_db(db_name):
    client = MongoClient("mongodb://localhost:27017/")
    db = client[db_name]

    return db


def query_movies_by_title(db_name, title):
    db = connect_to_db(db_name)
    collection_movies = db["movies"]

    found_movies_cursor = (
        collection_movies.find({"$text": {"$search": title}})
        .sort({"score": {"$meta": "textScore"}})
        .limit(10)
    )

    # for movie in found_movies_cursor:
    #     print("Index: {} | Title: {}".format(movie["index"], movie["title"]))

    return found_movies_cursor


def load_sparse_ratings(db_name):
    db = connect_to_db(db_name)

    collection_ratings_data = db["ratings_data"]
    collection_column_indices = db["ratings_column_indices"]
    collection_row_indices = db["ratings_row_indices"]

    ratings_data_cursor = collection_ratings_data.find({})

    ratings_data = ratings_data_cursor[0]["data"]
    n_users = ratings_data_cursor[0]["n_users"]
    n_movies = ratings_data_cursor[0]["n_movies"]

    column_indices_cursor = collection_column_indices.find({})
    column_indices = column_indices_cursor[0]["data"]

    row_indices_cursor = collection_row_indices.find({})
    row_indices = row_indices_cursor[0]["data"]

    sparse_ratings = sparse.csr_matrix(
        (ratings_data, (row_indices, column_indices)), shape=(n_movies, n_users)
    )

    return sparse_ratings


def get_data_column_row_indices(db_name):
    db = connect_to_db(db_name)

    collection_ratings_data = db["ratings_data"]
    collection_column_indices = db["ratings_column_indices"]
    collection_row_indices = db["ratings_row_indices"]

    ratings_data_cursor = collection_ratings_data.find({})
    ratings_data = ratings_data_cursor[0]["data"]

    column_indices_cursor = collection_column_indices.find({})
    column_indices = column_indices_cursor[0]["data"]

    row_indices_cursor = collection_row_indices.find({})
    row_indices = row_indices_cursor[0]["data"]

    return ratings_data, column_indices, row_indices


def get_n_users_and_n_movies(db_name):
    db = connect_to_db(db_name)

    collection_ratings_data = db["ratings_data"]
    ratings_data_cursor = collection_ratings_data.find({}, {"data": 0})
    n_users = ratings_data_cursor[0]["n_users"]
    n_movies = ratings_data_cursor[0]["n_movies"]

    return {"n_users": n_users, "n_movies": n_movies}


def get_intercept_latent_factors(db_name):
    n_users = get_n_users_and_n_movies(db_name)["n_users"]
    intercept_latent_factors = np.zeros((1, n_users))

    db = connect_to_db(db_name)
    collection_intercept_latent_factors = db["intercept_latent_factors"]
    intercept_latent_factors_cursor = collection_intercept_latent_factors.find({})

    for lt in intercept_latent_factors_cursor:
        intercept_latent_factors[:, lt["index"]] = lt["data"]

    return intercept_latent_factors


def get_users_latent_factors(db_name, lt_length):
    n_users = get_n_users_and_n_movies(db_name)["n_users"]
    users_latent_factors = np.zeros((n_users, lt_length))

    db = connect_to_db(db_name)
    collection_users_latent_factors = db["users_latent_factors"]
    users_latent_factors_cursor = collection_users_latent_factors.find({})

    for lt in users_latent_factors_cursor:
        users_latent_factors[lt["index"], :] = np.array(lt["data"])

    return users_latent_factors


def get_movies_latent_factors(db_name, lt_length):
    n_movies = get_n_users_and_n_movies(db_name)["n_movies"]
    movies_latent_factors = np.zeros((n_movies, lt_length))

    db = connect_to_db(db_name)
    collection_movies_latent_factors = db["movies_latent_factors"]
    movies_latent_factors_cursor = collection_movies_latent_factors.find({})

    for lt in movies_latent_factors_cursor:
        movies_latent_factors[lt["index"], :] = np.array(lt["data"])

    return movies_latent_factors


def get_intercept_latent_factor(db_name, index):
    db = connect_to_db(db_name)
    collection_intercept_latent_factors = db["intercept_latent_factors"]

    intercept_latent_factor_cursor = collection_intercept_latent_factors.find(
        {"index": index}
    )

    return intercept_latent_factor_cursor[0]


def get_movie_latent_factor(db_name, index):
    db = connect_to_db(db_name)
    collection_movies_latent_factors = db["movies_latent_factors"]

    movie_latent_factor_cursor = collection_movies_latent_factors.find({"index": index})

    return movie_latent_factor_cursor[0]


def get_user_latent_factor(db_name, index):
    db = connect_to_db(db_name)
    collection_users_latent_factors = db["users_latent_factors"]

    user_latent_factor_cursor = collection_users_latent_factors.find({"index": index})

    return user_latent_factor_cursor[0]


def get_user_rated_movies(db_name, user_id):
    db = connect_to_db(db_name)

    ratings_matrix = load_sparse_ratings(db_name).toarray()
    user_ratings = ratings_matrix[:, user_id]

    rated_movies_indices = np.nonzero(user_ratings)[0]
    rated_user_ratings = user_ratings[rated_movies_indices]

    collection_movies = connect_to_db("cinematch")["movies"]

    rated_movies_cursor = collection_movies.find(
        {"index": {"$in": rated_movies_indices.tolist()}}
    )

    rated_movies = []

    for rating, rated_movie in list(zip(rated_user_ratings, rated_movies_cursor)):
        genres = rated_movie["genres"].split("|")
        genres = ", ".join(genres)

        rated_movies.append(
            {
                "title": rated_movie["title"],
                "poster_url": (
                    rated_movie["poster_url"]
                    if type(rated_movie["poster_url"]) != float
                    else ""
                ),
                "genres": genres,
                "rating": rating,
                "index": rated_movie["index"],
            }
        )

    return rated_movies


def get_rated_movies_indices(db_name, user_id):
    ratings_matrix = load_sparse_ratings(db_name).toarray()
    user_ratings = ratings_matrix[:, user_id]
    rated_movies_indices = np.nonzero(user_ratings)

    return rated_movies_indices


def get_movie_info(db_name, movie_id):
    db = connect_to_db(db_name)
    collection_movies = db["movies"]
    movie_cursor = collection_movies.find({"index": movie_id})
    return movie_cursor[0]


def get_mean_movies_rating(db_name):
    db = connect_to_db(db_name)
    collection_mean_movies_rating = db["mean_rating"]
    mean_movie_rating_cursor = collection_mean_movies_rating.find({})
    return mean_movie_rating_cursor[0]["means"]


def get_predictions_user(db_name, user_id):
    db = connect_to_db(db_name)
    collection_predictions = db["predictions"]
    predictions_cursor = collection_predictions.find({"index": user_id})
    return predictions_cursor[0]["predictions"]


def update_n_users(db_name, new_n_users):
    db = connect_to_db(db_name)
    collection_ratings_data = db["ratings_data"]
    collection_ratings_data.update_one({}, {"$set": {"n_users": new_n_users}})


def save_document(db_name, collection, document):
    db = connect_to_db(db_name)
    db[collection].insert_one(document)


def save_new_rating_data(db_name, new_rating_data):
    db = connect_to_db(db_name)

    for rating in new_rating_data:
        db["ratings_data"].update_one({}, {"$push": {"data": rating[0]}})
        db["ratings_row_indices"].update_one({}, {"$push": {"data": rating[1]}})
        db["ratings_column_indices"].update_one({}, {"$push": {"data": rating[2]}})


def get_genres(db_name):
    db = connect_to_db(db_name)

    return db["genres"].find({})[0]
