from pydantic import BaseModel
from api.models.movie import Movie
from typing import List


class MovieList(BaseModel):
    movies: List[Movie]
