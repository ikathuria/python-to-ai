import pandas as pd
import cosine_similarity


movie_database = pd.DataFrame({
    "MovieID": [1, 2, 3, 4, 5],
    "Year": [1995, 1995, 1995, 1995, 1995],
    "Title": ["Toy Story", "Jumanji", "Grumpier Old Men", "Waiting to Exhale", "Father of the Bride Part 2"],
    "Action": [0, 0, 0, 0, 0],
    "Adventure": [1, 1, 0, 0, 0],
    "Animation": [1, 0, 0, 0, 0],
    "Children's": [1, 1, 0, 0, 0],
    "Comedy": [1, 0, 1, 1, 1],
    "Crime": [0, 0, 0, 0, 0],
    "Documentary": [0, 0, 0, 0, 0],
    "Drama": [0, 0, 0, 1, 0],
    "Fantasy": [1, 1, 0, 0, 0],
    "Film-Noir": [0, 0, 0, 0, 0],
    "Horror": [0, 0, 0, 0, 0],
    "Musical": [0, 0, 0, 0, 0],
    "Mystery": [0, 0, 0, 0, 0],
    "Romance": [0, 0, 1, 1, 0],
    "Sci-Fi": [0, 0, 0, 0, 0],
    "Thriller": [0, 0, 0, 0, 0],
    "War": [0, 0, 0, 0, 0],
    "Western": [0, 0, 0, 0, 0],
})

movie_genres = movie_database.filter(movie_database.columns[2:])
movie_genres.index = movie_genres[movie_genres.columns[0]]
movie_genres = movie_genres.drop(movie_genres.columns[0], axis=1).T

cos_sims = []
for m1 in movie_genres.columns:
    for m2 in movie_genres.columns:
        if m1 == m2:
            continue
        m1_g = movie_genres[m1].to_numpy()
        m2_g = movie_genres[m2].to_numpy()

        cos_sims.append([m1, m2, cosine_similarity(m1_g, m2_g)])

print(cos_sims)
