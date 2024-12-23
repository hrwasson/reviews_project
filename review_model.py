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
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score



#nltk.download( 'stopwords' )

sheet_id = '15kPBQi8fW6EV2P1k0MPN0iRictK9Vg5AnEKVojkKsRQ'
df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")
df.rename(columns={'Latitude': 'lat'}, inplace=True)
df.rename(columns={'Longitude': 'lon'}, inplace=True)

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
        return sentiment.describe(review)
    return "neutral"

def get_would_go_back(data): 
    scale = {
        1: "definitely would not go back", 
        2: "unlikely to go back", 
        3: "neutral to go back", 
        4: "likely to go back", 
        5: "definitely would go back"
    }
    return scale.get(data, "Unknown") 


df['Target'] = df['Would go back?'].map(get_would_go_back).astype('category')
df['Overall Sentiment'] = df['Positive Review'].fillna('') + " " + df['Negative Review'].fillna('')
cleaned_sentiment = df['Overall Sentiment'].apply(clean)  # Clean the text
df['Overall Sentiment'] = cleaned_sentiment.apply(get_overall_sentiment)  # Get sentiment terms

# setting x and y variables
# y doesn't include season for now
y = df['Target']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

x = df[['Category', 'Rating', 'Overall Sentiment', 'Atmosphere', 
       'Food quality', 'Service',
       'Unique Aspects', 'Parking', 'WiFi',
       'Charging Outlets', 'Price', 'Parking Type 1',
       'Parking Type 2']]

x = x.fillna('Missing')

x['Category'] = x['Category'].astype('category')
x['Overall Sentiment'] = x['Overall Sentiment'].astype('category')
x['Rating'] = x['Rating'].astype('category')
x['Atmosphere'] = x['Atmosphere'].astype('category')
x['Food quality'] = x['Food quality'].astype('category')
x['Service'] = x['Service'].astype('category')
x['Unique Aspects'] = x['Unique Aspects'].astype('category')
x['Parking'] = x['Parking'].astype('category')
x['WiFi'] = x['WiFi'].astype('category')
x['Charging Outlets'] = x['Charging Outlets'].astype('category')
x['Price'] = x['Price'].astype('category')
x['Parking Type 1'] = x['Parking Type 1'].astype('category')
x['Parking Type 2'] = x['Parking Type 2'].astype('category')

#categorical_features = x.select_dtypes(include=['category', 'object']).columns
x = pd.get_dummies(x)

# Encoding target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Check label distribution before splitting
print("Label distribution before splitting:")
print(pd.Series(y).value_counts())

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.25, random_state=444, stratify=y_encoded
)

nn_reviews = MLPClassifier(
    solver='lbfgs', alpha=1e-4,
    hidden_layer_sizes=(8,),
    random_state=444
)

nn_reviews.fit(X_train, y_train)

# Predict on test set
y_pred = nn_reviews.predict(X_test)

# Decode predictions and test labels
y_test_decoded = le.inverse_transform(y_test)
y_pred_decoded = le.inverse_transform(y_pred)

# Evaluation
print("Classification Report:")
print(classification_report(y_test_decoded, y_pred_decoded))

# Additional metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test_decoded, y_pred_decoded))