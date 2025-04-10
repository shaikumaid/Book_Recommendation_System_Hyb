#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Importing the required libraries and loading the data.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

Ratings_df=pd.read_excel("Ratings.xlsx")
Books_df=pd.read_excel("Books.xlsx")
Users_df=pd.read_excel("Users.xlsx")


# In[4]:


# Getting info on each dataset after loading
Ratings_df.info()
Users_df.info()
Books_df.info()


# In[5]:


# Shape of the dataframes
print("Ratings shape:", Ratings_df.shape)
print("Users shape:", Users_df.shape)
print("Books shape:", Books_df.shape)


# In[6]:


# Looking for missing values
print("\nMissing Values in Books:\n", Books_df.isnull().sum())
print("\nMissing Values in Ratings:\n", Ratings_df.isnull().sum())
print("\nMissing Values in Users:\n", Users_df.isnull().sum())


# In[7]:


# Look for duplicates and Drop if necessary
print("Duplicate Books:", Books_df.duplicated().sum())
print("Duplicate Ratings:", Ratings_df.duplicated().sum())
print("Duplicate Users:", Users_df.duplicated().sum())


Books_df.drop_duplicates(inplace=True)
Ratings_df.drop_duplicates(inplace=True)
Users_df.drop_duplicates(inplace=True)


# In[8]:


#Descriptive Statistics for each dataset
print(Books_df.describe(include='all'))
print(Ratings_df.describe(include='all'))
print(Users_df.describe(include='all'))


# # Books Dataset
# 

# In[9]:


top_authors = Books_df['Book-Author'].value_counts().head(10)

plt.figure(figsize=(9,4))
sns.barplot(x=top_authors.values, y=top_authors.index, color="skyblue")
plt.title("Top 10 Authors with Most Books")
plt.xlabel("Number of Books")
plt.ylabel("Author")
plt.tight_layout()
plt.show()


# In[10]:


top_books = Books_df['Book-Title'].value_counts().head(10)

plt.figure(figsize=(9,4))
sns.barplot(x=top_books.values, y=top_books.index, color="lightgreen")
plt.title("Top 10 Most Common Book Titles")
plt.xlabel("Number of Occurrences")
plt.ylabel("Book Title")
plt.tight_layout()
plt.show()


# In[11]:


top_publishers = Books_df['Publisher'].value_counts().head(10)

plt.figure(figsize=(9,4))
sns.barplot(x=top_publishers.values, y=top_publishers.index, color="orange")
plt.title("Top 10 Publishers with Most Books")
plt.xlabel("Number of Books")
plt.ylabel("Publisher")
plt.tight_layout()
plt.show()


# In[12]:


# Filter for years
valid_years = Books_df[Books_df["Year-Of-Publication"] > 1400]

books_per_year = valid_years["Year-Of-Publication"].value_counts().sort_index()


plt.figure(figsize=(11,4))
sns.lineplot(x=books_per_year.index, y=books_per_year.values, color="#87CEFA")
plt.title("Books Published Over the Years", fontsize=13)
plt.xlabel("Year", fontsize=11)
plt.ylabel("Number of Books", fontsize=11)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[13]:


books_filtered = Books_df[(Books_df["Year-Of-Publication"] >= 1000) & 
                          (Books_df["Year-Of-Publication"] <= 2020)]

books_per_year = books_filtered["Year-Of-Publication"].value_counts().sort_index()

plt.figure(figsize=(16,6))
sns.barplot(x=books_per_year.index.astype(int), y=books_per_year.values)
plt.title("Books Published Each Year", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Books Published", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# # Users Dataset

# In[14]:


# Fillling null values in the age colunm with the median
Users_df['Age'].fillna(Users_df['Age'].median(), inplace=True)
# Filtering out outliers in age (e.g., ages > 100 or < 5)
Users_df = Users_df[(Users_df['Age'] >= 5) & (Users_df['Age'] <= 100)]



# In[15]:


# Filter valid ages
valid_users = Users_df[(Users_df['Age'] >= 5) & (Users_df['Age'] <= 100)]

# Age distribution
plt.figure(figsize=(12,6))
sns.histplot(valid_users['Age'], bins=20, kde=False, color="#87CEFA", edgecolor='black')
plt.title("User Age Distribution", fontsize=16)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Number of Users", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# In[16]:


# Spliting the Location column into seperate columns
location_split = Users_df['Location'].str.split(',', expand=True)

# Create new columns
Users_df['City'] = location_split[0].str.strip()
Users_df['State'] = location_split[1].str.strip() if location_split.shape[1] > 1 else None
Users_df['Country'] = location_split[2].str.strip() if location_split.shape[1] > 2 else None

# Preview
Users_df[['Location', 'City', 'State', 'Country']].head()


# In[17]:


Users_df.info()


# In[18]:


Users_df.shape


# In[19]:


#Adding age filter
valid_users = Users_df[(Users_df['Age'] >= 5) & (Users_df['Age'] <= 100)]

# After splitting the Location, drop any null values in country
valid_users = valid_users.dropna(subset=['Country'])

# Get top 10 countries by number of users
top_countries = valid_users['Country'].value_counts().head(10).index
filtered_users = valid_users[valid_users['Country'].isin(top_countries)]

# Plot count of users per age for each country
plt.figure(figsize=(14,8))
sns.countplot(data=filtered_users, x='Country', hue='Age', palette='coolwarm')
plt.title("User Count per Age by Country", fontsize=16)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Number of Users", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="Age", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# # Ratings Dataset

# In[20]:


#Rating in values for each rating

# Drop missing values
Ratings_df = Ratings_df.dropna(subset=['Book-Rating'])

# Round ratings to nearest 0.5
Ratings_df['Count for each rating'] = (Ratings_df['Book-Rating'] * 2).round() / 2

# Count ratings in 0.5 intervals
rating_counts = Ratings_df['Count for each rating'].value_counts().sort_index()

print(rating_counts)


# In[21]:


# Drop missing ratings
Ratings_df = Ratings_df.dropna(subset=['Book-Rating'])

# Plot
plt.figure(figsize=(9,4))
sns.histplot(Ratings_df['Book-Rating'], binwidth=0.5, color="skyblue", edgecolor="black")

plt.title("Book Ratings", fontsize=13)
plt.xlabel("Rating (intervals of 0.5)", fontsize=11)
plt.ylabel("Count of Ratings", fontsize=1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# # Recommendation System
# 

# In[22]:


from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Sample data
data = Dataset.load_builtin('ml-100k')
algo = SVD()

# Run basic 3-fold cross validation
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)



# In[23]:


get_ipython().system('pip install scikit-surprise')


# In[35]:


#  Import Libraries
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, KNNWithMeans
from surprise.model_selection import train_test_split
from IPython.display import display, HTML

# Step 1: Data Preparation
# Merge Ratings and Books data
merged_df = Ratings_df.merge(Books_df, on='ISBN')
merged_df = merged_df.dropna(subset=['User-ID', 'Book-Rating', 'ISBN'])
merged_df['User-ID'] = merged_df['User-ID'].astype(int)
merged_df = merged_df[merged_df['Book-Rating'] > 0]

# Filter users with enough interactions
user_counts = merged_df['User-ID'].value_counts()
active_users = user_counts[user_counts >= 30].index
filtered_df = merged_df[merged_df['User-ID'].isin(active_users)]

# Surprise expects rating_scale and required columns
reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(filtered_df[['User-ID', 'ISBN', 'Book-Rating']], reader)
trainset = data.build_full_trainset()

# Train User-Based KNN Model
sim_options = {'name': 'pearson_baseline', 'user_based': True}
user_model = KNNBasic(k=100, min_k=5, sim_options=sim_options)
user_model.fit(trainset)

top_books = filtered_df['ISBN'].value_counts().head(10000).index
reduced_df = filtered_df[filtered_df['ISBN'].isin(top_books)]

# Prepare Surprise dataset from reduced_df
reader = Reader(rating_scale=(0, 10))
item_data = Dataset.load_from_df(reduced_df[['User-ID', 'ISBN', 'Book-Rating']], reader)
item_trainset = item_data.build_full_trainset()


# Train item-based model with cosine similarity
sim_options = {'name': 'cosine', 'user_based': False}
item_model = KNNWithMeans(k=30, sim_options=sim_options, verbose=True)
item_model.fit(item_trainset)

def display_similar_books(book_title, n=5):
    if book_title not in title_to_isbn:
        print("❌ Book title not found.")
        return

    isbn = title_to_isbn[book_title]
    try:
        inner_id = trainset_item.to_inner_iid(isbn)
        neighbors = item_model.get_neighbors(inner_id, k=n)
        neighbor_isbns = [trainset_item.to_raw_iid(inner_id) for inner_id in neighbors]
    except ValueError:
        print("⚠️ Book not found in training set.")
        return

    html = f"<h2>📖 Books similar to <i>{book_title}</i></h2>"

    for neighbor_isbn in neighbor_isbns:
        book = Books_df[Books_df['ISBN'] == neighbor_isbn]
        if not book.empty:
            book = book.iloc[0]
            avg_rating = reduced_df[reduced_df['ISBN'] == neighbor_isbn]['Book-Rating'].mean()
            predicted_score = item_model.predict(0, neighbor_isbn).est  # 0 is a placeholder user
            html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <img src="{book['Image-URL-M']}" style="height: 150px; margin-right: 15px;">
                <div>
                    <b>{book['Book-Title']}</b><br>
                    Author: {book['Book-Author']}<br>
                    Predicted Score: {predicted_score:.2f}<br>
                    Average Rating: {avg_rating:.2f}
                </div>
            </div>
            """

    display(HTML(html))

# After model training or before hybrid_recommend
def display_fallback_books(n=5):
    top_books_df = (
        filtered_df.groupby('ISBN')
        .agg({'Book-Rating': ['count', 'mean']})
        .reset_index()
    )
    top_books_df.columns = ['ISBN', 'RatingCount', 'AvgRating']
    top_books_df = top_books_df[top_books_df['RatingCount'] >= 50]
    top_books_df = top_books_df.sort_values(by='AvgRating', ascending=False).head(n)

    html = f"<h2>📚 Popular Books You Might Like</h2>"

    for _, row in top_books_df.iterrows():
        book = Books_df[Books_df['ISBN'] == row['ISBN']]
        if not book.empty:
            book = book.iloc[0]
            html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <img src="{book['Image-URL-M']}" style="height: 150px; margin-right: 15px;">
                <div>
                    <b>{book['Book-Title']}</b><br>
                    Author: {book['Book-Author']}<br>
                    Average Rating: {row['AvgRating']:.2f}
                </div>
            </div>
            """
    display(HTML(html))


def hybrid_recommend(user_input, n=5):
    if isinstance(user_input, int) or user_input.isdigit():
        user_id = int(user_input)
        if user_id in filtered_df['User-ID'].unique():
            display_user_recommendations(user_id, n)
        else:
            print("⚠️ User ID not found. Try a book title instead.")
    else:
        book_title = user_input.strip()
        if book_title in title_to_isbn:
            isbn = title_to_isbn[book_title]
            if isbn in item_trainset._raw2inner_id_items:
                display_similar_books(book_title, n)
            else:
                print("⚠️ This book exists in our catalog, but we don't have enough ratings to give similar recommendations.")
                print("📖 Try another popular book, or search using a User ID for personalized suggestions.")
                display_fallback_books(n)
        else:
            print("❌ Book not found in catalogue.")
            print("📚 Please check the spelling or try a different title.")


# In[37]:


# Get all raw ISBNs (book IDs) from the item-based training set
train_isbns = [trainset_item.to_raw_iid(inner_id) for inner_id in trainset_item.all_items()]

# Map ISBNs to Titles (assuming you have this dictionary)
isbn_to_title = dict(zip(Books_df['ISBN'], Books_df['Book-Title']))

# Get book titles from the trained ISBNs
train_titles = [isbn_to_title[isbn] for isbn in train_isbns if isbn in isbn_to_title]

# Print how many books are trained
print(f"📚 Total trained books: 10000")

# Show a few sample titles
for title in train_titles[:25]:
    print(title)


# In[25]:


# Step 1: ISBNs in training set
train_isbns = set(reduced_df['ISBN'].unique())

# Step 2: All ISBNs in full dataset
all_isbns = set(filtered_df['ISBN'].unique())

# Step 3: Books not in training set
untrained_isbns = list(all_isbns - train_isbns)

# Step 4: Show sample books not in training
untrained_books = Books_df[Books_df['ISBN'].isin(untrained_isbns)].drop_duplicates(subset='ISBN')
print(f"❌ Total books not in training set: {len(untrained_books)}")

# Show a few as sample
untrained_books[['Book-Title', 'Book-Author']].head(10)


# In[26]:


# Get all raw ISBNs (book IDs) from the trainset
train_isbns = [trainset.to_raw_iid(inner_id) for inner_id in trainset.all_items()]
print(f"Total books in training set: {len(train_isbns)}")


# In[27]:


#  Import Libraries
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from IPython.display import display, HTML

# Step 1: Data Preparation
# Merge Ratings and Books data
merged_df = Ratings_df.merge(Books_df, on='ISBN')
merged_df = merged_df.dropna(subset=['User-ID', 'Book-Rating', 'ISBN'])
merged_df['User-ID'] = merged_df['User-ID'].astype(int)
merged_df = merged_df[merged_df['Book-Rating'] > 0]

# Filter users with enough interactions
user_counts = merged_df['User-ID'].value_counts()
active_users = user_counts[user_counts >= 30].index
filtered_df = merged_df[merged_df['User-ID'].isin(active_users)]

# Surprise expects rating_scale and required columns
reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(filtered_df[['User-ID', 'ISBN', 'Book-Rating']], reader)
trainset = data.build_full_trainset()

# Train User-Based KNN Model
sim_options = {'name': 'pearson_baseline', 'user_based': True}
user_model = KNNBasic(k=100, min_k=5, sim_options=sim_options)
user_model.fit(trainset)

top_books = filtered_df['ISBN'].value_counts().head(5000).index
reduced_df = filtered_df[filtered_df['ISBN'].isin(top_books)]

# Prepare Surprise dataset from reduced_df
reader = Reader(rating_scale=(0, 10))
item_data = Dataset.load_from_df(reduced_df[['User-ID', 'ISBN', 'Book-Rating']], reader)
trainset_item = item_data.build_full_trainset()

# Train item-based model with cosine similarity
sim_options = {'name': 'cosine', 'user_based': False}
item_model = KNNWithMeans(k=30, sim_options=sim_options, verbose=True)
item_model.fit(item_trainset)

# Create lookup dictionaries
title_to_isbn = dict(zip(Books_df['Book-Title'].str.strip(), Books_df['ISBN']))
isbn_to_title = dict(zip(Books_df['ISBN'], Books_df['Book-Title'].str.strip()))


def display_similar_books(book_title, n=5):
    if book_title not in title_to_isbn:
        print("❌ Book title not found.")
        return

    isbn = title_to_isbn[book_title]
    try:
        inner_id = trainset_item.to_inner_iid(isbn)
        neighbors = item_model.get_neighbors(inner_id, k=n)
        neighbor_isbns = [trainset_item.to_raw_iid(inner_id) for inner_id in neighbors]
    except ValueError:
        print("⚠️ Book not found in training set.")
        return

    html = f"<h2>📖 Books similar to <i>{book_title}</i></h2>"

    for neighbor_isbn in neighbor_isbns:
        book = Books_df[Books_df['ISBN'] == neighbor_isbn]
        if not book.empty:
            book = book.iloc[0]
            avg_rating = reduced_df[reduced_df['ISBN'] == neighbor_isbn]['Book-Rating'].mean()
            predicted_score = item_model.predict(0, neighbor_isbn).est  # 0 is a placeholder user
            html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <img src="{book['Image-URL-M']}" style="height: 150px; margin-right: 15px;">
                <div>
                    <b>{book['Book-Title']}</b><br>
                    Author: {book['Book-Author']}<br>
                    Predicted Score: {predicted_score:.2f}<br>
                    Average Rating: {avg_rating:.2f}
                </div>
            </div>
            """

    display(HTML(html))

# After model training or before hybrid_recommend
def display_fallback_books(n=5):
    top_books_df = (
        filtered_df.groupby('ISBN')
        .agg({'Book-Rating': ['count', 'mean']})
        .reset_index()
    )
    top_books_df.columns = ['ISBN', 'RatingCount', 'AvgRating']
    top_books_df = top_books_df[top_books_df['RatingCount'] >= 50]
    top_books_df = top_books_df.sort_values(by='AvgRating', ascending=False).head(n)

    html = f"<h2>📚 Popular Books You Might Like</h2>"

    for _, row in top_books_df.iterrows():
        book = Books_df[Books_df['ISBN'] == row['ISBN']]
        if not book.empty:
            book = book.iloc[0]
            html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <img src="{book['Image-URL-M']}" style="height: 150px; margin-right: 15px;">
                <div>
                    <b>{book['Book-Title']}</b><br>
                    Author: {book['Book-Author']}<br>
                    Average Rating: {row['AvgRating']:.2f}
                </div>
            </div>
            """
    display(HTML(html))


def hybrid_recommend(user_input, n=5):
    if isinstance(user_input, int) or user_input.isdigit():
        user_id = int(user_input)
        if user_id in filtered_df['User-ID'].unique():
            display_user_recommendations(user_id, n)
        else:
            print("⚠️ User ID not found. Try a book title instead.")
    else:
        book_title = user_input.strip()
        if book_title in title_to_isbn:
            isbn = title_to_isbn[book_title]
            if isbn in item_trainset._raw2inner_id_items:
                display_similar_books(book_title, n)
            else:
                print("⚠️ This book exists in our catalog, but we don't have enough ratings to give similar recommendations.")
                print("📖 Try another popular book, or search using a User ID for personalized suggestions.")
                display_fallback_books(n)
        else:
            print("❌ Book not found in catalogue.")
            print("📚 Please check the spelling or try a different title.")


# In[28]:


#DEMO#
#Predictive Score – personalized predicted rating based on similarity.
#Average Rating – global average rating of the book from the dataset.

print("✨ Welcome to the Book Recommendation System!")
print("You can enter either a User ID or a Book Title to get recommendations.\n")
user_input = input("🔍 Enter a User ID or Book Title: ")
hybrid_recommend(user_input)


# In[29]:


#DEMO#
#Predictive Score – personalized predicted rating based on similarity.
#Average Rating – global average rating of the book from the dataset.

print("✨ Welcome to the Book Recommendation System!")
print("You can enter either a User ID or a Book Title to get recommendations.\n")
user_input = input("🔍 Enter a User ID or Book Title: ")
hybrid_recommend(user_input)


# In[30]:


#DEMO#
#Predictive Score – personalized predicted rating based on similarity.
#Average Rating – global average rating of the book from the dataset.

print("✨ Welcome to the Book Recommendation System!")
print("You can enter either a User ID or a Book Title to get recommendations.\n")
user_input = input("🔍 Enter a User ID or Book Title: ")
hybrid_recommend(user_input)


# In[ ]:




