"""
For presentation:

What is the length of the data in letter or characters?

"""

import json

DATA = r"4 Deep Learning\NLP\Baby_GPT\data\imdb_detailed.json"
SAVE_PATH = r"4 Deep Learning\NLP\Baby_GPT\data\processed_imdb_dataset.txt"

with open(DATA, "r", encoding="utf-8") as f:
	imdb_data = json.load(f)

def preprocess_movies(movies, output_file='movies_for_gpt.txt'):
    lines = []

    for movie in movies:
        movie_lines = []
        movie_lines.append(f"### MOVIE START ###")
        movie_lines.append(f"Title: {movie.get('title', 'N/A')}")
        movie_lines.append(f"Year: {movie.get('release_year', 'N/A')}")
        movie_lines.append(f"Type: {movie.get('type', 'N/A')}")
        movie_lines.append(f"PG Rating: {movie.get('pg_rating', 'N/A')}")
        movie_lines.append(f"IMDb Rating: {movie.get('rating', 'N/A')}")
        movie_lines.append(f"Summary: {movie.get('summary', 'N/A')}")
        movie_lines.append(f"Creators: {', '.join(movie.get('creators', []))}")

        # Stars
        stars_list = [
            f"{s['actor']} as {s['character']}" for s in movie.get('stars', [])]
        movie_lines.append(f"Cast: {', '.join(stars_list)}")

        # Keywords
        movie_lines.append(f"Keywords: {', '.join(movie.get('keywords', []))}")

        # Reviews
        reviews_text = []
        for r in movie.get('top_reviews', []):
            reviews_text.append(
            	f"[Rating: {r.get('rating', 'N/A')}] {r.get('title', '')}: {r.get('content', '')}")
        movie_lines.append(f"Reviews: {' | '.join(reviews_text)}")

        # Similar Items
        similar_text = [f"{item['title']} ({item.get('link', '')})" for item in movie.get(
            'similar_items', [])]
        movie_lines.append(f"Similar Items: {', '.join(similar_text)}")

        movie_lines.append(f"### MOVIE END ###\n")

        lines.append('\n'.join(movie_lines))

    # Write to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Advanced dataset ready! Lines:", len(lines))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Output saved to {output_file}")


# Run preprocessing
preprocess_movies(imdb_data, output_file=SAVE_PATH)
