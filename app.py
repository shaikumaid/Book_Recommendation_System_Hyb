import streamlit as st
st.set_page_config(page_title="📚 Hybrid Book Recommender", layout="wide")

import pandas as pd
from joblib import load
from difflib import get_close_matches

import sys
st.write(f"🧪 Python version: {sys.version}")


# --- LOAD DATA ---
@st.cache_data
def load_data():
    books = pd.read_excel("Books.xlsx")
    ratings = pd.read_excel("Ratings.xlsx")
    df = pd.merge(ratings, books, on='ISBN')
    df = df[df['Book-Rating'] > 0]
    df.dropna(subset=['Book-Title', 'Book-Author', 'Image-URL-M'], inplace=True)

    # ✅ Limit to top 1000 most-rated books
    top_books = df['ISBN'].value_counts().head(1000).index
    df = df[df['ISBN'].isin(top_books)]
    
    return books, ratings, df

Books_df, Ratings_df, filtered_df = load_data()

# --- LOAD PRETRAINED MODELS ---
user_knn = load("user_model.joblib")
knn_model = load("item_model.joblib")
trainset = load("trainset.joblib")

# --- POPULAR BOOKS FALLBACK ---
def get_popular_books(n=10):
    top_books = filtered_df.groupby('Book-Title').agg({
        'Book-Rating': ['mean', 'count']
    }).reset_index()
    top_books.columns = ['Book-Title', 'Avg-Rating', 'Count']
    top_books = top_books[top_books['Count'] > 20].sort_values(by='Avg-Rating', ascending=False).head(n)
    return Books_df[Books_df['Book-Title'].isin(top_books['Book-Title'])].drop_duplicates('Book-Title')


# --- HYBRID RECOMMENDER ---
def hybrid_recommendation(input_text):
    results = []

    # Check if input is user ID
    if input_text.isdigit():
        user_id = int(input_text)
        if user_id not in Ratings_df['User-ID'].unique():
            return get_fallback()

        all_books = filtered_df['ISBN'].unique()
        predictions = []
        for isbn in all_books:
            try:
                pred = user_knn.predict(user_id, isbn)
                predictions.append((isbn, pred.est))
            except:
                continue

        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]
        for isbn, score in predictions:
            book = Books_df[Books_df['ISBN'] == isbn].drop_duplicates('Book-Title')
            if not book.empty:
                row = book.iloc[0]
                avg_rating = filtered_df[filtered_df['ISBN'] == isbn]['Book-Rating'].mean()
                results.append(format_result(row, score, avg_rating))
        return results

    # Else assume input is book title
    matches = get_close_matches(input_text, Books_df['Book-Title'].unique(), n=1, cutoff=0.6)
    if not matches:
        return get_fallback()
    
    matched_title = matches[0]
    input_isbn = Books_df[Books_df['Book-Title'] == matched_title]['ISBN'].values[0]
    try:
        inner_id = trainset.to_inner_iid(input_isbn)
        neighbors = knn_model.get_neighbors(inner_id, k=10)
        for inner in neighbors:
            isbn = trainset.to_raw_iid(inner)
            book = Books_df[Books_df['ISBN'] == isbn].drop_duplicates('Book-Title')
            if not book.empty:
                row = book.iloc[0]
                pred = knn_model.predict(0, isbn).est  # Dummy user
                avg_rating = filtered_df[filtered_df['ISBN'] == isbn]['Book-Rating'].mean()
                results.append(format_result(row, pred, avg_rating))
    except:
        return get_fallback()

    return results


def get_fallback():
    fallback = get_popular_books()
    results = []
    for _, row in fallback.iterrows():
        avg_rating = filtered_df[filtered_df['Book-Title'] == row['Book-Title']]['Book-Rating'].mean()
        results.append(format_result(row, "N/A", avg_rating))
    return results


def format_result(row, pred, avg):
    return {
        'title': row['Book-Title'],
        'author': row['Book-Author'],
        'image': row['Image-URL-M'],
        'predicted': round(pred, 2) if pred != "N/A" else "N/A",
        'avg_rating': round(avg, 2)
    }

# --- STREAMLIT UI ---
st.title("📚 Hybrid Book Recommendation System")

user_input = st.text_input("Enter a User ID or Book Title:", placeholder="e.g. 276725 or Harry Potter")

if user_input:
    st.write("🔍 Starting recommendation...")
    recs = hybrid_recommendation(user_input)

    for rec in recs:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(rec['image'], use_column_width=True)
        with col2:
            st.markdown(f"**{rec['title']}**  \n*by {rec['author']}*")
            st.markdown(f"⭐ Predicted: {rec['predicted']} | Avg: {rec['avg_rating']}")
        st.markdown("---")
