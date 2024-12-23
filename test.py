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
nltk.download( 'stopwords' )

sheet_id = '15kPBQi8fW6EV2P1k0MPN0iRictK9Vg5AnEKVojkKsRQ'
df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")
df['id'] = df.index + 1

# creating two data frames to separate out the positive and negative reviews wiht the associated ID
positive = df[['id', 'Name', 'Positive Review']]
negative = df[['id', 'Name', 'Negative Review']]

# test string
df['Overall Review'] = df['Positive Review'] + " " + df['Negative Review']

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

def get_overall_sentiment(review): 
    if sentiment.exist(review): 
        sentiment_overall = sentiment.describe(review)
        return sentiment_overall

print(get_overall_sentiment(clean(df['Overall Review'][0])))

