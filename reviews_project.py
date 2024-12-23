import nltk.stem.porter
import pandas as pd
import numpy as np
from pandasql import sqldf
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import re
import nltk
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentiment_module import sentiment_term, sentiment
import collections
import nltk
import re
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display
from transformers import pipeline
from wordcloud import WordCloud
from wordcloud import STOPWORDS
nltk.download( 'stopwords' )


# importing data from google sheets
sheet_id = '15kPBQi8fW6EV2P1k0MPN0iRictK9Vg5AnEKVojkKsRQ'
df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")
df['id'] = df.index + 1

# creating two data frames to separate out the positive and negative reviews wiht the associated ID
positive = df[['id', 'Name', 'Positive Review']]
negative = df[['id', 'Name', 'Negative Review']]

# test string
test_pos = positive['Positive Review'][2]
test_n = negative['Negative Review'][3]

# function creation
def clean(review): 
    stop_words = nltk.corpus.stopwords.words( 'english' )
    term_list = []
    review = str(review)
    for term in review.split():
        if term not in stop_words: 
            term = term.lower()
            term_list.append(term)  # Append term to the list
    return term_list  # Return the list after the loop finishes

def get_sentiment(review): 
    sentiment_list = []
    for term in review: 
        if sentiment.exist(term):
            sent_term = sentiment.describe(term)
            sentiment_list.append(sent_term)
    return np.unique(sentiment_list)

def get_overall_sentiment(review): 
    if sentiment.exist(review): 
        sentiment_overall = sentiment.describe(review)
    return sentiment_overall

def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

def search(title): 
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indicies = np.argpartition(similarity, -5)[-5:]
    results = df.iloc[indicies]
    return results[::-1]


def word_cloud(data): 

    sentiment_pipeline = pipeline("sentiment-analysis")

    review = data['Overall Review'].to_list()

    # Run sentiment analysis
    result = sentiment_pipeline(review)

    result_df = pd.DataFrame(result)

    result_df['Overall Review'] = data['Overall Review']
    
    # Wordcloud with positive tweets
    positive_reviews = result_df['Overall Review'][result_df['label'] == 'POSITIVE']
    stop_words = nltk.corpus.stopwords.words( 'english' )
    positive_wordcloud = WordCloud(max_font_size=50,
                                    max_words=100, 
                                    background_color="white",
                                    stopwords = stop_words, 
                                    colormap='PRGn').generate(str(positive_review))
    plt.figure()
    plt.imshow(positive_wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Goal: find 5 words in each review (positive, negative) to describe the place 
# Remove stop words from term vectors

positive_data = []
negative_data = []
for i in range(len(df)): 
    positive_review = get_overall_sentiment(clean(df['Positive Review'][i]))
    positive_data.append(positive_review)

    negative_review = get_overall_sentiment(clean(df['Negative Review'][i]))
    negative_data.append(negative_review)

p_df = pd.DataFrame(positive_data)
n_df = pd.DataFrame(negative_data)

p_df['id'] = df['id']
n_df['id'] = df['id']

p_df = p_df.rename(columns={'0': 'positive sentiment'}, inplace=True)
n_df = n_df.rename(columns={'0': 'negative sentiment'}, inplace=True)

df['Review'] = df['Positive Review'] + ' ' + df['Negative Review']

# cleaning the title of all the locations
df['New Review'] = df['Review'].apply(clean_title)

# creating a TFIDF matrix of word frequencies
vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf = vectorizer.fit_transform(df['New Review'])

def find_similar_locations(id): 
    similar_locations = df[(df['id'] == id) & (df['Rating'] > 3.5)]["id"].unique()
    similar_locations_recs = df[(df['id'] == id) & (df['Rating'] > 3.5)]["id"]
    # finding only the movies 10% or more of places I liked
    similar_locations_recs = similar_locations_recs.value_counts() / len(similar_locations)
    similar_locations_recs = similar_locations_recs[similar_locations_recs > 0.1]

    all_me = df[(df['id'].isin(similar_locations_recs.index) & df['Rating'] > 3.5)]

    # big differential locations are similar to the input locatioin instead of just similar to that location
    all_me_recs = all_me['id'].value_counts()/len(all_me['id'].unique())

    rec_percentages = pd.concat([similar_locations_recs, all_me_recs], axis=1)
    rec_percentages.columns = ["similar locations", "all locations"]


    # higher the score the better the recommendations is 
    rec_percentages['score'] = rec_percentages['similar locations']/rec_percentages['all locations']
    rec_percentages = rec_percentages.sort_values("score", ascending = False)

    return rec_percentages.head(10).merge(df, left_index=True, right_on='id')


input_name = widgets.Text(
    value = "", 
    description = "Category:", 
    disabled = False
)

recommendation_list = widgets.Output()

def on_type(data): 
    with recommendation_list:
        recommendation_list.clear_output()
        title = data['new']
        if len(title) > 3:
            results = search(title)
            id = results.iloc[0]['id']
            display(find_similar_locations(id))

input_name.observe(on_type, names = 'value')
display(input_name, recommendation_list)

