from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load pre-processed data
movies_df = pd.read_csv("pre_processing_done.csv")

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

# Stemming Function
porter = PorterStemmer()
def stem(text):
    return " ".join([porter.stem(word) for word in text.split()])

movies_df["tags"] = movies_df["tags"].apply(stem)
vectors = vectorizer.fit_transform(movies_df["tags"]).toarray()
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie_title):
    try:
        index = movies_df[movies_df['title'].str.lower() == movie_title.lower()].index[0]
        distances = similarity[index]
        movie_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:7]
        recommendations = [movies_df.iloc[i[0]].title for i in movie_list]
        return recommendations
    except:
        return ["Movie not found! Please try again."]

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    if request.method == "POST":
        movie_title = request.form["movie_title"]
        recommendations = recommend(movie_title)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)

