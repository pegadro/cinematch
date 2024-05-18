from fastapi import APIRouter
from db import db
from fastapi.responses import JSONResponse
import random
import numpy as np
import pandas as pd
from api.models.movieList import MovieList

router = APIRouter()
DB_NAME = "cinematch_classification"


@router.get("/rated_movies/{user_id}")
async def get_rated_movies(user_id: int):
    movies = db.get_user_rated_movies(DB_NAME, int(user_id))
    random.shuffle(movies)
    # print(movies)
    return JSONResponse(status_code=200, content=movies)


@router.get("/user_recommendations/{user_id}")
async def get_user_recommendations(user_id: int):
    rated_movies_indices = db.get_rated_movies_indices(DB_NAME, user_id)[0]

    predictions = np.array(db.get_predictions_user(DB_NAME, user_id))

    users_movies = db.get_n_users_and_n_movies(DB_NAME)
    n_users = users_movies["n_users"]
    n_movies = users_movies["n_movies"]

    bl_ratings_matrix = np.full((n_movies, n_users), -1)
    ratings_data, column_indices, row_indices = db.get_data_column_row_indices(DB_NAME)
    bl_ratings_matrix[row_indices, column_indices] = ratings_data

    ratings_count = np.count_nonzero(bl_ratings_matrix != -1, axis=1)
    positive_ratings_count = np.count_nonzero(bl_ratings_matrix == 1, axis=1)

    ix = np.flip(np.argsort(predictions))

    positive_proportion = positive_ratings_count / ratings_count
    positive_proportion[np.isnan(positive_proportion)] = 0

    movies = pd.DataFrame(
        data={
            "index": np.arange(len(ratings_count)),
            "ratings": ratings_count,
            "positive_proportion": positive_proportion,
            "pred": predictions,
        }
    )

    movies_to_recommend = (
        movies.loc[ix[:600]]
        .loc[movies["ratings"] > 50]
        .sort_values("positive_proportion", ascending=False)
    )
    movies_to_recommend = movies_to_recommend[
        ~movies_to_recommend["index"].isin(rated_movies_indices)
    ]

    recommendations = []
    for i in range(len(movies_to_recommend)):
        movie = db.get_movie_info("cinematch", movies_to_recommend.iloc[i]["index"])

        genres = movie["genres"].split("|")
        genres = ", ".join(genres)

        recommendations.append(
            {
                "title": movie["title"],
                "poster_url": (
                    movie["poster_url"] if type(movie["poster_url"]) != float else ""
                ),
                "genres": genres,
                "prediction": movies_to_recommend.iloc[i]["pred"],
                "index": movie["index"],
            }
        )

    random.shuffle(recommendations)
    return recommendations
