# Movie-Multi-Language-Recommendation-System
This project is a Movie Recommendation System designed to suggest similar movies based on genre, popularity, rating, and most importantly, the user's preferred language. It uses unsupervised learning (KMeans Clustering) to group movies with similar features and recommends films from the same cluster filtered by language.

🚀 Features
Genre-based KMeans clustering to find similar movies.

Language filtering to recommend content in the user's preferred language (e.g., Hindi, Telugu).

Ranking recommendations by vote average and popularity.

Clean data preprocessing with genre and language parsing.

Built using Python, pandas, scikit-learn, and matplotlib in Google Colab.

📂 Dataset
Used dataset: Indian Movies.csv

Key features: title, genres, vote_average, popularity, spoken_languages

🧠 Technologies & Libraries
pandas, numpy, ast, scikit-learn, seaborn, matplotlib

Google Colab for development

💡 Example Output
Input: Movie = Dangal, Language = Hindi

Title	Vote Average	Popularity
Chak De India	8.3	92.4
Sultan	7.9	89.7
Bhaag Milkha Bhaag	8.0	85.3
Mary Kom	7.6	79.5
Lagaan	8.1	76.8

🔮 Future Scope
Add filtering by actors, directors, or release year.

Integrate user feedback and personalization using collaborative filtering.

Use deep learning for more dynamic recommendations.


![image](https://github.com/user-attachments/assets/96a22d74-2edb-40d7-917d-aaf08c0ee795)

