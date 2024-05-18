from pydantic import BaseModel


class Movie(BaseModel):
    title: str
    poster_url: str
    genres: str
    index: int
    rating: float
