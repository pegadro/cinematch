from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import random
import pandas as pd
import numpy as np
from db import db
from sklearn.linear_model import Ridge

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Movie(BaseModel):
    title: str
    poster_url: str
    genres: str
    index: int
    rating: float

class MovieList(BaseModel):
    movies: List[Movie]

@app.get("/")
async def hello_world():
    return JSONResponse(status_code=200, content={"message": "hola mundo"})

@app.get("/rated_movies/{user_id}")
async def get_rated_movies(user_id: int):
    movies = db.get_user_rated_movies(int(user_id))
    random.shuffle(movies)
    return JSONResponse(status_code=200, content=movies)

@app.get("/user_recommendations/{user_id}")
async def get_user_recommendations(user_id: int):
    rated_movies_indices = db.get_rated_movies_indices(user_id)[0]
        
    predictions = np.array(db.get_predictions_user(user_id))
    
    ratings_matrix = db.load_sparse_ratings().toarray()
    
    ratings_count = np.count_nonzero(ratings_matrix, axis=1)
    
    ix = np.flip(np.argsort(predictions))
    
    means = db.get_mean_movies_rating()
    
    movies = pd.DataFrame(data={
        "index": np.arange(len(ratings_count)),
        "ratings": ratings_count,
        "mean_rating": means,
        "pred": predictions
    })
    
    movies_to_recommend = movies.loc[ix[:600]].loc[movies["ratings"] > 50].sort_values("mean_rating", ascending=False)
    movies_to_recommend = movies_to_recommend[~movies_to_recommend["index"].isin(rated_movies_indices)]
    
    recommendations = []
    for i in range(len(movies_to_recommend)):
        movie = db.get_movie_info(movies_to_recommend.iloc[i]["index"])
        
        genres = movie["genres"].split("|")
        genres = ", ".join(genres)
        
        recommendations.append({
            "title": movie["title"],
            "poster_url": movie["poster_url"] if type(movie["poster_url"]) != float else "",
            "genres": genres,
            "prediction": movies_to_recommend.iloc[i]["pred"],
            "index": movie["index"]
        })
        
    random.shuffle(recommendations)
    return recommendations
    
@app.get("/search_movie/{title}")
async def search_movie_by_title(title: str):
    search = db.query_movies_by_title(title)
    
    results = []
    for result in search:
        
        genres = result["genres"].split("|")
        genres = ", ".join(genres)
        
        results.append({
            "title": result["title"],
            "poster_url": result["poster_url"] if type(result["poster_url"]) != float else "",
            "genres": genres,
            "index": result["index"]
        })
        
    return results

@app.post("/new_user/")
async def create_new_user(initialRatings: MovieList):
    # Learn latent factors for user
    n_movies = db.get_n_users_and_n_movies()["n_movies"]
    n_users = db.get_n_users_and_n_movies()["n_users"]
    
    new_user_ratings = np.zeros(n_movies)
    
    for movie in initialRatings.movies:
        new_user_ratings[movie.index] = movie.rating
        
    rated_movies_indices = np.nonzero(new_user_ratings)[0]
    rated_movies_latent_factors = np.zeros((len(initialRatings.movies), 600))
    
    for i, index in enumerate(rated_movies_indices):
        rated_movies_latent_factors[i,:] = db.get_movie_latent_factor(int(index))["data"]
    
    means = np.array(db.get_mean_movies_rating())
    target = new_user_ratings[new_user_ratings != 0]
    
    target_norm = target - means[new_user_ratings != 0]
    
    model = Ridge(alpha=1.5)
    model.fit(rated_movies_latent_factors, target_norm)
    
    learned_user_latent_factor = model.coef_
    learned_intercept_latent_factor = model.intercept_
    
    # test
    rated_movies_indices = np.nonzero(new_user_ratings)[0]

    for index in rated_movies_indices:
        movie_latent_factor = db.get_movie_latent_factor(int(index))["data"]
        mean = means[index]
        prediction = np.dot(np.array(movie_latent_factor), learned_user_latent_factor.T) + learned_intercept_latent_factor
        prediction += mean
        print("{}. Rated: {} - Predicted: {}".format(index, new_user_ratings[index], prediction))
        
    # upload new user data
    new_user_index = n_users
    
    # intercept
    db.save_document("intercept_latent_factors", {
        "index": new_user_index,
        "data": learned_intercept_latent_factor
    })
    
    # user lt
    
    db.save_document("users_latent_factors", {
        "index": new_user_index,
        "data": learned_user_latent_factor.tolist()
    })
    
    # save predictions
    movies_lt = db.get_movies_latent_factors()
    
    predictions = np.matmul(movies_lt, learned_user_latent_factor.T) + learned_intercept_latent_factor
    predictions += means
    
    db.save_document("predictions", {
        "index": new_user_index,
        "predictions": predictions.tolist()
    })
    
    # save ratings
    new_rating_data = [ (movie.rating, movie.index, new_user_index) for movie in initialRatings.movies ]
    db.save_new_rating_data(new_rating_data)
    
    db.update_n_users(n_users + 1)
    
@app.get("/genres/")
async def get_genres():
    return db.get_genres()["data"]