import pandas as pd
movies = pd.read_csv(r'F:\Lambton_Classes\AML\python_project\ml-25m\movies1.csv')
print(movies)


import re
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)
	
	
movies["clean_title"] = movies["title"].apply(clean_title)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf = vectorizer.fit_transform(movies["clean_title"])


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


#Search Algorithm
def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5]
    results = movies.iloc[indices][::-1]
    return results
	
	
import ipywidgets as widgets
from IPython.display import display


movie_input = widgets.Text(
    value="Toy Story",
    description="Movie Title:",
    disabled=False
)
movie_list = widgets.Output()


def on_type(data):
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            display(search(title))
			
			
movie_input.observe(on_type, names='value')
display(movie_input, movie_list)


ratings = pd.read_csv(r'F:\Lambton_Classes\AML\python_project\ml-25m\ratings1.csv')

ratings.dtypes
movie_id = 1


def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .10]

    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    
    rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

	
movie_name_input = widgets.Text(
    value="Toy Story",
    description="Movie Title",
    disabled=False
)

recommendation_list = widgets.Output()

def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            display(find_similar_movies(movie_id))
        
movie_name_input.observe(on_type, names="value")

display(movie_name_input, recommendation_list)