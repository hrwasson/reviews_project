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
import streamlit as st
from haversine import haversine, Unit
import networkx as nx
import geopy.distance
from transformers import pipeline
from wordcloud import WordCloud
from wordcloud import STOPWORDS
nltk.download( 'stopwords' )


# importing data from google sheets
sheet_id = '15kPBQi8fW6EV2P1k0MPN0iRictK9Vg5AnEKVojkKsRQ'
df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")
df.rename(columns={'Latitude': 'lat'}, inplace=True)
df.rename(columns={'Longitude': 'lon'}, inplace=True)
df.sort_values(by = 'Timestamp', inplace=True)
df['Count of Reviews'] = 1
df['Overall Review'] = df['Positive Review'] + " " + df['Negative Review']

sentiment_pipeline = pipeline("sentiment-analysis")


test = df['Positive Review'][0]
test2 = df['Overall Review'][0]

review = df['Overall Review'].to_list()

# Run sentiment analysis
result = sentiment_pipeline(review)

result_df = pd.DataFrame(result)

result_df['Overall Review'] = df['Overall Review']
 
# Wordcloud with positive tweets
positive_tweets = result_df['Overall Review'][result_df['label'] == 'POSITIVE']
stop_words = ["https", "co", "RT"] + list(STOPWORDS)
positive_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", 
                               colormap='PRGn',stopwords = stop_words).generate(str(positive_tweets))
plt.figure()
plt.title("Positive Words - Wordcloud")
plt.imshow(positive_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

st.pyplot(positive_wordcloud)
 
# Wordcloud with negative tweets
# negative_tweets = df['tweet'][df["sentiment"] == 'NEG']
# stop_words = ["https", "co", "RT"] + list(STOPWORDS)
# negative_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords = stop_words).generate(str(negative_tweets))
# plt.figure()
# plt.title("Negative Tweets - Wordcloud")
# plt.imshow(negative_wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()
