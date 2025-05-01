import pandas as pd
import numpy as np
import ast  # For parsing JSON-like strings
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Load the movie dataset
dataset_path = ("Indian Movies.csv")  # Set your actual dataset path
df = pd.read_csv(dataset_path, low_memory=False)

# Select relevant columns
movies = df[['title', 'genres', 'vote_average', 'popularity', 'spoken_languages']].copy()

# Drop missing values
movies = movies.dropna(subset=['genres', 'vote_average', 'popularity', 'spoken_languages'])

# Function to parse genres correctly
def parse_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [g['name'] for g in genres if 'name' in g]
    except (ValueError, SyntaxError):
        return []

movies['genre_list'] = movies['genres'].apply(parse_genres)

# Function to parse spoken languages
def parse_spoken_languages(languages_str, preferred_language):
    try:
        if isinstance(languages_str, str):
            languages = ast.literal_eval(languages_str)
            return any(language.get('iso_639_1') == preferred_language for language in languages)
        return False
    except (ValueError, SyntaxError):
        return False

# Function to filter movies based on language preference
def filter_movies_by_language(movies, preferred_language):
    return movies[movies['spoken_languages'].apply(lambda x: parse_spoken_languages(x, preferred_language))]

# Convert genre lists into binary features
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(movies['genre_list'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

# Normalize vote_average and popularity
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(movies[['vote_average', 'popularity']])
scaled_df = pd.DataFrame(scaled_features, columns=['vote_average', 'popularity'])

# Combine all features
final_features = pd.concat([genre_df, scaled_df], axis=1)

# Apply KMeans clustering
optimal_k = 5  
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
movies['cluster'] = kmeans.fit_predict(final_features)

# Movie recommendation function
def recommend_movies(movie_title, preferred_language='en', n=5):
    movie_title = movie_title.strip().lower()
    target = movies[movies['title'].str.strip().str.lower() == movie_title]
    
    if target.empty:
        available_titles = movies['title'].str.strip().str.lower().tolist()
        return f"❌ Movie titled '{movie_title}' not found. Available titles: {available_titles}"

    filtered_movies = filter_movies_by_language(movies, preferred_language)
    cluster_id = target['cluster'].values[0]

    similar_movies = filtered_movies[(filtered_movies['cluster'] == cluster_id) & 
                                     (filtered_movies['title'].str.strip().str.lower() != movie_title)]

    if similar_movies.empty:
        return f"⚠ No similar movies found for '{movie_title}' in '{preferred_language}' language."

    return similar_movies[['title', 'vote_average', 'popularity']].sort_values(
        by=['vote_average', 'popularity'], ascending=False).head(n)

# Example usage
print("\n🎬 Recommended movies similar to 'RRR' in Telugu:")
print(recommend_movies("RRR", preferred_language='te', n=5))

print("\n🎬 Recommended movies similar to 'Dangal' in Hindi:")
print(recommend_movies("Dangal", preferred_language='hi', n=5))
