
import streamlit as st
import pandas as pd
import pickle
from surprise import KNNBasic, KNNWithMeans

# Load required data and models
Books_df = pickle.load(open("Books_df.pkl", "rb"))
filtered_df = pickle.load(open("filtered_df.pkl", "rb"))
title_to_isbn = pickle.load(open("title_to_isbn.pkl", "rb"))
user_model = pickle.load(open("user_model.pkl", "rb"))
item_model = pickle.load(open("item_model.pkl", "rb"))
trainset = pickle.load(open("trainset.pkl", "rb"))
trainset_item = pickle.load(open("trainset_item.pkl", "rb"))

def display_user_recommendations(user_id, n=5):
    all_isbns = filtered_df['ISBN'].unique()
    rated_isbns = filtered_df[filtered_df['User-ID'] == user_id]['ISBN'].unique()
    unseen_isbns = list(set(all_isbns) - set(rated_isbns))

    predictions = [user_model.predict(user_id, isbn) for isbn in unseen_isbns]
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    st.markdown(f"<h3>📚 Top {n} Recommendations for User ID <code>{user_id}</code></h3>", unsafe_allow_html=True)

    for rec in top_predictions:
        book = Books_df[Books_df['ISBN'] == rec.iid]
        if not book.empty:
            book = book.iloc[0]
            avg_rating = filtered_df[filtered_df['ISBN'] == rec.iid]['Book-Rating'].mean()
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <img src="{book['Image-URL-M']}" style="height: 150px; margin-right: 15px;">
                    <div>
                        <b>{book['Book-Title']}</b><br>
                        Author: {book['Book-Author']}<br>
                        Predicted Score: {rec.est:.2f}<br>
                        Average Rating: {avg_rating:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

def display_similar_books(book_title, n=5):
    if book_title not in title_to_isbn:
        st.error("❌ Book title not found.")
        return

    isbn = title_to_isbn[book_title]
    try:
        inner_id = trainset_item.to_inner_iid(isbn)
        neighbors = item_model.get_neighbors(inner_id, k=n)
        neighbor_isbns = [trainset_item.to_raw_iid(inner_id) for inner_id in neighbors]
    except ValueError:
        st.warning("⚠️ Not enough ratings for this book.")
        return

    st.markdown(f"<h3>📖 Books similar to <i>{book_title}</i></h3>", unsafe_allow_html=True)

    for neighbor_isbn in neighbor_isbns:
        book = Books_df[Books_df['ISBN'] == neighbor_isbn]
        if not book.empty:
            book = book.iloc[0]
            avg_rating = filtered_df[filtered_df['ISBN'] == neighbor_isbn]['Book-Rating'].mean()
            predicted_score = item_model.predict(0, neighbor_isbn).est
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <img src="{book['Image-URL-M']}" style="height: 150px; margin-right: 15px;">
                    <div>
                        <b>{book['Book-Title']}</b><br>
                        Author: {book['Book-Author']}<br>
                        Predicted Score: {predicted_score:.2f}<br>
                        Average Rating: {avg_rating:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

# Streamlit App UI
st.title("📚 Book Recommendation System (RS3)")
st.markdown("Enter a **User ID** or a **Book Title** to get recommendations:")

user_input = st.text_input("🔍 Your input:")

if st.button("Get Recommendations"):
    if user_input.isdigit():
        user_id = int(user_input)
        if user_id in filtered_df['User-ID'].unique():
            display_user_recommendations(user_id)
        else:
            st.warning("⚠️ User ID not found. Try a book title.")
    else:
        title = user_input.strip()
        if title in title_to_isbn:
            isbn = title_to_isbn[title]
            if isbn in trainset_item._raw2inner_id_items:
                display_similar_books(title)
            else:
                st.warning("⚠️ Book found in metadata but lacks ratings.")
        else:
            st.error("❌ Book title not found. Try again.")
