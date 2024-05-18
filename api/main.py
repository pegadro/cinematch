from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from db import db

from api.routers import regression, classification

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(regression.router, prefix="/regression")
app.include_router(classification.router, prefix="/classification")


@app.get("/genres/")
async def get_genres():
    return db.get_genres("cinematch")["data"]


@app.get("/search_movie/{title}")
async def search_movie_by_title(title: str):
    search = db.query_movies_by_title("cinematch", title)

    results = []
    for result in search:

        genres = result["genres"].split("|")
        genres = ", ".join(genres)

        results.append(
            {
                "title": result["title"],
                "poster_url": (
                    result["poster_url"] if type(result["poster_url"]) != float else ""
                ),
                "genres": genres,
                "index": result["index"],
            }
        )

    return results
