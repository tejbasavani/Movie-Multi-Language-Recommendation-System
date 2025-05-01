import pandas as pd
import numpy as np
import ast  # Helps parse genres stored as string lists
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv("Movies.csv")

# Step 2: Convert genres from string to actual lists
df["Genres"] = df["Genres"].apply(ast.literal_eval)

# Step 3: Encode genres using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df["Genres"])

# Step 4: Convert encoded genres into DataFrame
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
df = pd.concat([df, genre_df], axis=1)  # Merge encoded genres back to main dataset

# Step 5: Scale features for clustering
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df.iloc[:, 3:])  # Exclude title/description

# Step 6: Apply KMeans Clustering
num_clusters = 3  # Set based on desired number of groups
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_features)

# Step 7: Evaluate clustering quality (Silhouette Score)
score = silhouette_score(scaled_features, df["Cluster"])
print(f"Silhouette Score: {score:.3f}")

# Step 8: Visualize Clusters
sns.scatterplot(x=df["Sci-Fi"], y=df["Drama"], hue=df["Cluster"], palette="viridis")
plt.title("Movie Clusters Based on Genre")
plt.xlabel("Sci-Fi Score")
plt.ylabel("Drama Score")
plt.show()

# Step 9: Automatic Tagging
df["Generated Tags"] = df["Genres"].apply(lambda x: ", ".join(x) + " Movie")
print(df[["Movie Title", "Generated Tags"]])

import pandas as pd
import numpy as np
import ast  # for parsing genre strings
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = "Movies.csv"
df = pd.read_csv(file_path)

# Fix minor issues (Correct movie title spelling)
df.loc[df["Title"] == "Amlie", "Title"] = "Amélie"

# Correct genre formatting
df["Genres"] = df["Genres"].apply(lambda x: x.split(", "))

# Apply MultiLabelBinarizer for genres
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df["Genres"])
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# Normalize numerical features
scaler = MinMaxScaler()
df[["Popularity", "Vote Average"]] = scaler.fit_transform(df[["Popularity", "Vote Average"]])

# Combine processed data
df_final = pd.concat([df[["Movie ID", "Title", "Popularity", "Vote Average", "Language"]], genres_df], axis=1)

# Determine number of clusters dynamically
num_clusters = min(3, len(df_final) - 1)  # Avoid silhouette score errors with small datasets

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df_final["Cluster"] = kmeans.fit_predict(df_final.drop(columns=["Movie ID", "Title", "Language"]))

# Evaluate clustering quality if dataset size allows
if len(df_final) > 2:
    sil_score = silhouette_score(df_final.drop(columns=["Movie ID", "Title", "Language", "Cluster"]), df_final["Cluster"])
    print(f"Silhouette Score: {sil_score:.3f}")
else:
    print("Silhouette score skipped due to insufficient samples.")

# Function to recommend movies based on language preference
def recommend_movies(language, df_final, top_n=5):
    filtered_movies = df_final[df_final["Language"] == language].sort_values(by="Popularity", ascending=False).head(top_n)
    return filtered_movies[["Title", "Language", "Popularity", "Vote Average"]]

# Example usage
language_preference = "English"
recommended_movies = recommend_movies(language_preference, df_final)
print("\nRecommended Movies:")
print(recommended_movies)

# Visualization: Popularity vs Vote Average
plt.figure(figsize=(8,5))
sns.scatterplot(x=df_final["Popularity"], y=df_final["Vote Average"], hue=df_final["Language"], palette="Set2")
plt.title("Popularity vs Vote Average (Grouped by Language)")
plt.xlabel("Popularity")
plt.ylabel("Vote Average")
plt.legend(title="Language")
plt.show()
