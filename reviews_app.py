import pandas as pd
import nltk
import nltk.stem.porter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentiment_module import sentiment_term, sentiment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import tensorflow
import transformers
from transformers import pipeline
from wordcloud import WordCloud
from wordcloud import STOPWORDS
nltk.download( 'stopwords' )
nltk.download('punkt')
import datetime
import geopy
import re
import usaddress

st.set_page_config(layout="wide")

# Access the sheet_id
sheet_id = st.secrets["sheets"]["url"]

print(f"Sheet ID: {sheet_id}")

df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")

df.rename(columns={'Latitude': 'lat'}, inplace=True)
df.rename(columns={'Longitude': 'lon'}, inplace=True)
df.sort_values(by = 'Timestamp', inplace=True)
df['Count of Reviews'] = 1
df['Overall Review'] = df['Positive Review'] + " " + df['Negative Review']

# getting the colors for the ratings 
 
def get_color(rating):
    if rating == 1: return '#efbbff'
    elif rating == 2: return '#d896ff'
    elif rating == 3: return '#be29ec'
    elif rating == 4: return '#800080'
    elif rating == 5: return '#660066'

# getting the sizes for the ratings
def get_size(rating):
    if rating == 1: return 1
    elif rating == 2: return 2
    elif rating == 3: return 3
    elif rating == 4: return 4
    elif rating == 5: return 5
    
def get_price(price): 
    if price == 1: return '$'
    if price == 2: return '$$'
    if price == 3: return '$$$'
    if price == 4: return '$$$$'
    
def parking(data):
    return data[(data['Parking'] == 4)| (data['Parking'] == 5)]
    
def wifi(data): 
    return data[data['WiFi'] == 'Yes']
    
def charging_outlets(data): 
    return data[data['Charging Outlets'] == 'Yes']
    
def format_lat_long(name, lat, lon): 
    return (name, (lat, lon))

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
    
def word_cloud(data, output_path = "word_cloud.png"): 

    sentiment_pipeline = pipeline("sentiment-analysis")
    review = data['Positive Review'].to_list()
    result = sentiment_pipeline(review)
    result_df = pd.DataFrame(result)
    result_df['Positive Review'] = data['Positive Review']
    positive_review = result_df['Positive Review'][(result_df['label'] == 'POSITIVE') | (result_df['label'] == 'NEGATIVE')][0]
    stop_words = nltk.corpus.stopwords.words( 'english' )
    positive_wordcloud = WordCloud(max_font_size=60,
                                    max_words=100, 
                                    background_color="white",
                                    stopwords = stop_words, 
                                    colormap='BuPu').generate(str(positive_review))
    plt.figure()
    plt.imshow(positive_wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path

def word_cloud_other(data, output_path = "word_cloud.png"): 

    sentiment_pipeline = pipeline("sentiment-analysis")
    review = data['Positive Review'].to_list()
    result = sentiment_pipeline(review)
    result_df = pd.DataFrame(result)
    result_df['Positive Review'] = data['Positive Review']
    positive_review = result_df['Positive Review'][(result_df['label'] == 'POSITIVE') | (result_df['label'] == 'NEGATIVE')][1:6]
    stop_words = nltk.corpus.stopwords.words( 'english' ) 
    positive_wordcloud = WordCloud(max_font_size=60,
                                    max_words=100, 
                                    background_color="white",
                                    stopwords = stop_words, 
                                    colormap='BuPu').generate(str(positive_review))
    plt.figure()
    plt.imshow(positive_wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path

def parking_colors(parking_type): 
    if parking_type == "Street": return '#fcfbfd'
    elif parking_type == "Parking Lot": return '#efedf5'
    elif parking_type == "Parking Garage": return '#dadaeb'
    elif parking_type == "Drive Through": return '#bcbddc'
    elif parking_type == "Uber Eats": return '#9e9ac8'
    elif parking_type == "Door Dash": return '#807dba'
    elif parking_type == "Metro": return '#6a51a3'
    elif parking_type == "None": return '#54278f'

def rating(data): 
    if data == 'Not for me': return 1
    if data == 'Fair': return 2
    if data == 'Okay but not great': return 3
    if data =='Really good': return 4
    if data == 'Best place ever!': return 5

def atmosphere(data): 
    if data == 'Uninviting': return 1
    if data == 'Not my vibe': return 2
    if data == 'Decent': return 3
    if data =='Almost perfect': return 4
    if data == 'Amazing': return 5

def food_quality(data):
    if data == 'Not my favorite': return 1
    if data == 'Mediocre': return 2
    if data == 'Decent': return 3
    if data =='Very good': return 4
    if data == 'Outstanding': return 5

def service(data):
    if data == 'Just okay': return 1
    if data == 'Could be better': return 2
    if data == 'Meets expectations': return 3
    if data =='Very friendly': return 4
    if data == 'Super Kind': return 5

def unique_aspects(data): 
    if data == 'Not my favorite': return 1
    if data == 'Mediocre': return 2
    if data == 'Decent': return 3
    if data =='Very good': return 4
    if data == 'Outstanding': return 5   

def go_back(data): 
    if data == 'Definitely Not': return 1
    if data == 'Probably Not': return 2
    if data == 'Maybe': return 3
    if data =='Probably Yes': return 4
    if data == 'Absolutely': return 5   

def recommendations(df):

    selected_region = st.selectbox("Where are you traveling to?", 
                 options = df['City'].dropna().unique(), 
                 index= None,
                 placeholder="Select a location")
    
    if selected_region: 
        filtered_activities = df[df['City'] == selected_region]['Category'].unique()
    else: 
        filtered_activities = []

    try: 
        selected_activity = st.selectbox("What kind of activity are you looking to do?", 
                 options = filtered_activities,
                 index=None,
                 placeholder='Select an activity'
                 )
    except: 
        print("idk what went wrong")

    if selected_activity: 

        reccomendation_df = df[(df['City'] == selected_region) & (df['Category'] == selected_activity)]

    
    start_price, end_price = st.select_slider(
        "Select a price range",
        options=[
            "$",
            "$$",
            "$$$",
            "$$$$"
        ],
        value=("$", "$$$$"),
        
    )

    price_df = df[(df['City'] == selected_region) & (df['Category'] == selected_activity) & (df['Price Range'] >= start_price) & (df['Price Range'] <= end_price)]

    ammenities_select = st.selectbox("What amenities are important to you? (e.g., parking, Wi-Fi, charging outlets, etc.)", 
                                      options = ['None', 'Easy Parking', 'Wi-Fi', 'Charging Outlets'],
                                      placeholder='None', 
                                      index=None
                                      )
    ammenities_df = price_df.copy()


    if ammenities_select == 'Easy Parking' : ammenities_df = parking(price_df)
    elif ammenities_select == 'Wi-Fi': ammenities_df = wifi(price_df)
    elif ammenities_select == 'Charging Outlets': ammenities_df = charging_outlets(price_df)  
    elif ammenities_select == 'None': ammenities_df = ammenities_df

    st.divider()

    # putting location suggestion here: 

    if not ammenities_df.empty:  # Check if filtered DataFrame has data

        tab1, tab2 = st.tabs(['Overall Recommendation', 'Other Locations You Might Enjoy'])

        with tab1: 
            ammenities_df['Rating'] = pd.to_numeric(ammenities_df['Rating'], errors='coerce')

            output = ammenities_df.sort_values(by=['Rating'], ascending=False).reset_index(drop=True)

            #text outputs
            emotion = output['Overall Sentiment'][0]
            name = output['Name'][0]
            location = output['Location'][0]
            wifi_present = output['WiFi'][0]
            charging = output['Charging Outlets'][0]
            price = output['Price Range'][0]
            parking1 = output['Parking Type 1'][0]
            output['Parking Type 2'].fillna('', inplace = True)
            parking2 = output['Parking Type 2'][0]
            whatIGot = output['What I got/did'][0]
            
            st.subheader(f"Based on your selections, you would be {emotion} at:")

            text_content = (
                    f"**Name:** {name}\n\n"
                    f"**Located at:** {location}\n\n"
                    f"**Price:** {price}\n\n"
                    f"**Amenities present:**\n"
                    f"- **Wi-Fi:** {wifi_present}\n"
                    f"- **Charging Outlets:** {charging}\n\n"
                    f"**You can expect to park:** {parking1} | {parking2}\n\n"
                    f"**Something you might enjoy there:** {whatIGot}")
            
            st.markdown(text_content)

            word_cloud_review = word_cloud(output)

            st.image(word_cloud_review)

            map_output = pd.DataFrame(output).head(1)

            reccomendation_map = px.scatter_mapbox(
                map_output,
                lat='lat',
                lon='lon',
                size='size',
                hover_name='Name',
                color='Rating',  # Use the Rating column for color mapping
                color_continuous_scale='Purpor',
                mapbox_style='carto-positron',
                title=f'{name}:',
                width=1000,
                height=500, 
                zoom = 15
            )
            reccomendation_map.update_traces(
                hovertemplate="<b>%{hovertext}</b>"  # Only show the name
            )
            st.plotly_chart(reccomendation_map)
        
        with tab2: 
            ammenities_df['Rating'] = pd.to_numeric(ammenities_df['Rating'], errors='coerce')

            output = ammenities_df.sort_values(by=['Rating'], ascending=False).reset_index(drop=True)
            
            #text outputs
            #emotion = output['Overall Sentiment'][1:6]

            if len(output) >= 2: 

                emotion = output['Overall Sentiment'][1]
                name = output['Name'][1]
                location = output['Location'][1]
                wifi_present = output['WiFi'][1]
                charging = output['Charging Outlets'][1]
                price = output['Price Range'][1]
                parking1 = output['Parking Type 1'][1]
                output['Parking Type 2'].fillna('', inplace = True)
                parking2 = output['Parking Type 2'][1]
                whatIGot = output['What I got/did'][1]
                
                #st.subheader(f"Based on your selections, you would be {emotion} at:")

                st.markdown(
                        f"**Name:** {name}\n\n"
                        f"**Located at:** {location}\n\n"
                        f"**Price:** {price}\n\n"
                        f"**Amenities present:**\n"
                        f"- **Wi-Fi:** {wifi_present}\n"
                        f"- **Charging Outlets:** {charging}\n\n"
                        f"**You can expect to park:** {parking1} | {parking2}\n\n"
                        f"**Something you might enjoy there:** {whatIGot}")
                
                map_output = pd.DataFrame(output).head(2)

                reccomendation_map = px.scatter_mapbox(
                    map_output,
                    lat='lat',
                    lon='lon',
                    size='size',
                    hover_name='Name',
                    color='Rating',  # Use the Rating column for color mapping
                    color_continuous_scale='Purpor',
                    mapbox_style='carto-positron',
                    title='Other Reviewed Locations',
                    width=1000,
                    height=500, 
                    zoom = 7
                )
                reccomendation_map.update_traces(
                    hovertemplate="<b>%{hovertext}</b>"  # Only show the name
                )
                st.plotly_chart(reccomendation_map)

            elif len(output) >= 4: 

                name1, name2, name3 = output['Name'][1:4]
                location1, location2, location3 = output['Location'][1:4]
                wifi_present1, wifi_present2, wifi_present3 = output['WiFi'][1:4]
                charging1, charging2, charging3 = output['Charging Outlets'][1:4]
                price1, price2, price3 = output['Price Range'][1:4]
                parking1, parking12, parking13 = output['Parking Type 1'][1:4]
                output['Parking Type 2'].fillna('', inplace = True)
                parking21, parking22, parking23 = output['Parking Type 2'][1:4]
                whatIGot1, whatIGot2, whatIGot3 = output['What I got/did'][1:4]
                
                st.subheader("Other locations you might enjoy...")

                st.markdown(
                    f"1. **Name:** {name1}\n\n"
                    f"**Located at:** {location1}\n\n"
                    f"**Price:** {price1}\n\n"
                    f"**Amenities present:**\n"
                    f"- **Wi-Fi:** {wifi_present1}\n"
                    f"- **Charging Outlets:** {charging1}\n\n"
                    f"**You can expect to park:** {parking1} | {parking21}\n\n"
                    f"**Something you might enjoy there:** {whatIGot1}"
                )

                st.divider()

                st.markdown(
                    f"2. **Name:** {name2}\n\n"
                    f"**Located at:** {location2}\n\n"
                    f"**Price:** {price2}\n\n"
                    f"**Amenities present:**\n"
                    f"- **Wi-Fi:** {wifi_present2}\n"
                    f"- **Charging Outlets:** {charging2}\n\n"
                    f"**You can expect to park:** {parking12} | {parking22}\n\n"
                    f"**Something you might enjoy there:** {whatIGot2}"
                )
                st.divider()
                st.markdown(
                    f"3. **Name:** {name3}\n\n"
                    f"**Located at:** {location3}\n\n"
                    f"**Price:** {price3}\n\n"
                    f"**Amenities present:**\n"
                    f"- **Wi-Fi:** {wifi_present3}\n"
                    f"- **Charging Outlets:** {charging3}\n\n"
                    f"**You can expect to park:** {parking13} | {parking23}\n\n"
                    f"**Something you might enjoy there:** {whatIGot3}"
                )

                word_cloud_review = word_cloud_other(output)

                st.image(word_cloud_review)

                map_output = pd.DataFrame(output).head(5)

                reccomendation_map = px.scatter_mapbox(
                    map_output,
                    lat='lat',
                    lon='lon',
                    size='size',
                    hover_name='Name',
                    color='Rating',  # Use the Rating column for color mapping
                    color_continuous_scale='Purpor',
                    mapbox_style='carto-positron',
                    title='Other Reviewed Locations',
                    width=1000,
                    height=500, 
                    zoom = 10
                )
                reccomendation_map.update_traces(
                    hovertemplate="<b>%{hovertext}</b>"  # Only show the name
                )
                st.plotly_chart(reccomendation_map)
            else: 
                st.write('Sorry there are no further recommendations. Please check back again soon. :)')

    else:
        st.write("No locations match your selected amenities. Please try a different option.")

def lat(address):
    if address is not None:
        locate = geopy.Nominatim(user_agent="Geopy Library")
        location = locate.geocode(address)
        return location.latitude

def lon(address):
    if address is not None:
        locate = geopy.Nominatim(user_agent="Geopy Library")
        location = locate.geocode(address)
        return location.longitude

def city(address):
    parsed = usaddress.parse(address)
    for val in parsed: 
        var1, var2 = val
        if var2 == 'PlaceName': 
            city = var1
            city = city.replace(",", "")
            return city
    
def state(address): 
    parsed = usaddress.parse(address)
    for val in parsed: 
        var1, var2 = val
        if var2 == 'StateName': 
            state = var1
            state = state.replace(",", "")
            return state

def spider_chart(df, attributes):
    df = pd.DataFrame(df)
    median_df = df.groupby('Category')[attributes].median().reset_index()
    
    radar_data = pd.melt(median_df, id_vars='Category', 
                         var_name='theta', value_name='r')
    
    fig = px.line_polar(radar_data, r='r', theta='theta', color = 'Category', line_close=True, color_discrete_sequence=px.colors.sequential.Plasma_r,
                template="plotly_dark", width= 700, height=500)
    
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])),showlegend=True)

    st.plotly_chart(fig, on_select=callable)

df['color'] = df['Rating'].apply(get_color)
df['size'] = df['Rating'].apply(get_size)
df['Rating'] = df['Rating'].to_list()
df['Price Range'] = df['Price'].apply(get_price)
df['Overall Sentiment'] = df['Overall Review'].apply(clean).apply(get_overall_sentiment)

#********************************************************************************************************************************************
# creating the streamlit app: 

#  setting page configuration 

# putting my picture on the sidebar

logo = "hannah.jpg"
st.sidebar.image(logo)

# Customize page title

# Sidebar navigation

st.markdown('''
            <div style="background-color:white; color:black">
            ''', unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "How did I collect the data?", "Contribute Reviews", "Data at a Glance", "My Recommendations", "Your Recommendations", "Give me feedback"], 
    index=0, 
    placeholder= "Where would you like to navigate to?"
)

markdown = '''

<div style="background-color:white; color:black; padding:15px; border-radius:10px;">
    <b>Hey there!</b> My name is <b>Hannah Wasson</b>, and I am a 2025 M.S. Candidate in Analytics at the 
    <b>Institute for Advanced Analytics, NC State University</b>. Since moving to Raleigh, NC, I’ve made it my 
    mission to track all the places I’ve visited to share them with my friends and family who live far away. 
    Along the way, I’ve honed my data collection techniques, practiced coding and model-building skills to analyze my reviews, 
    and discovered a passion for finding the best coffee shops to study at in the Triangle. Follow along as I continue exploring 
    new places and growing my data science expertise!
</div>
'''

# Sidebar content
st.sidebar.title("About Me")
st.sidebar.markdown(markdown, unsafe_allow_html=True)


if page == "Home":
    st.title("Eats & Adventures Tracker - Home")

    col1, col2 = st.columns(2)

    with col1: 

        rtp_map = df[df['City'].isin(['Cary', 'Raleigh', 'Durham', 'Morrisville'])]
        count_rtp = len(rtp_map)
        reccomendation_map = px.scatter_mapbox(
                        rtp_map,
                        lat='lat',
                        lon='lon',
                        size='size',
                        hover_name='Name',
                        #color='Count of Reviews',  # Use the Rating column for color mapping
                        #color_continuous_scale='Purpor',
                        color_discrete_sequence=['purple'],
                        mapbox_style='carto-positron',
                        title=f'{count_rtp} Reviews in Raleigh/Durham/Chapel Hill',
                        width=1000,
                        height=500, 
                        zoom = 8.5
                    )
        reccomendation_map.update_traces(
                        hovertemplate="<b>%{hovertext}</b>"  # Only show the name
                    )
        st.plotly_chart(reccomendation_map)

    with col2: 

        dc_map = df[df['City'].isin(['Arlington', 'Washington, D.C.'])]
        count_dc = len(dc_map)
        reccomendation_map = px.scatter_mapbox(
                        dc_map,
                        lat='lat',
                        lon='lon',
                        size='size',
                        hover_name='Name',
                        #color='Count of Reviews',  # Use the Rating column for color mapping
                        #color_continuous_scale='Purpor',
                        color_discrete_sequence=['purple'],
                        mapbox_style='carto-positron',
                        title=f'{count_dc} Reviews in Arlington, VA and Washington, D.C.',
                        width=1000,
                        height=500, 
                        zoom = 9
                    )
        reccomendation_map.update_traces(
                        hovertemplate="<b>%{hovertext}</b>"  # Only show the name
                    )

        st.plotly_chart(reccomendation_map)

    left, middle, right = st.columns((2, 7, 2))

    with middle:
        attribute_list = ['Food quality', 'Service', 'Unique Aspects', 'Atmosphere']
        spider_chart(df, attribute_list)


elif page == "How did I collect the data?":
    st.title("Eats & Adventures Tracker - How did I collect the data?")

    # Label field
    st.markdown(
        """
        I've been collecting the data via a Google Form, reviewing on the following characteristics:

        - Visit Information: Date of Visit
        - Location Details: Name, Category, Location, Latitude, Longitude, and Parking Ease and Types (e.g., Street Parking, Parking Lot).
        - Visit Context: Season Visited, What I Got/Did.
        - Ratings and Reviews: Overall Rating, Atmosphere, Food Quality, Service, Unique Aspects, Positive Review, Negative Review, and whether I would return to the place.
        - Amenities: WiFi Availability, and Charging Outlets
        - Cost: Price (e.g. $, $$, $$$, $$$$)    

        """
    )

    # Add the Google Form image
    st.image("datacollection.png")

elif page == "Contribute Reviews": 

    st.title("Eats & Adventures Tracker - Contribute Reviews")
    
    st.subheader('Please fill out the form with your reviews:')

    csv_file = "form_submission.csv"

    reviews_form = st.form("your_reviews", clear_on_submit=True, enter_to_submit=False)

    with reviews_form:
        st.subheader("Visit information")

        name = st.text_input('What is the name of the location?',
                              placeholder="Please fill in the official name of the location...")
        
        address = st.text_input('Where is it located?',
                                placeholder="e.g., 123 Main Street, City, State, ZIP")
        
        category  = st.selectbox(
            "What category does this location fall in?",
            ("Coffee Shop", "Restaurant", "Bar", "Dessert", 
             "Fast Food", "Diner", "Bakery", "Park", "Museum", 
             "Outdoor Activity", "Indoor Activity", "Uber Eats", 
             "Door Dash", "Brewery"),
            index=None,
            placeholder="Select a category...",
        )
        season = st.selectbox("What season did you visit in?", 
                                ("Summer", "Fall", "Winter", "Spring"),
                                index=None,
                                placeholder='Please select a season')
        
        order = st.text_area("What did you do or order at the location? If you did not order anything please leave blank.",
                             placeholder='Please fill in the box...')

        st.subheader("Rating information")

        r = st.selectbox(
            "What is your overall rating of this location?",
            options = ['Not for me', 'Fair', 'Okay but not great', 'Really good', 'Best place ever!'], 
            index=None,
            placeholder='Please select your overall rating...'
        )
        p = st.selectbox(
            "What is the price range?",
            options = ['$', '$$', '$$$', '$$$$'], 
            index=None, 
            placeholder="Please fill select the price range."
        )
        a = st.selectbox(
            "What is your rating on the atmosphere? This should reflect your opinion on the overall comfort and mood of the space.",
            options = ['Uninviting', 'Not my vibe', 'Decent', 'Almost perfect', 'Amazing'], 
            index=None, 
            placeholder="Please select your rating of the atmosphere..."
        )
        f = st.selectbox(
            "What is your rating on the food quality? This should reflect your opinion on the overall enjoyment of the food you ordered. Feel free to skip if you did not order food! ",
            options = ['Not my favorite', 'Mediocre', 'Decent', 'Very good', 'Outstanding'], 
             index=None, 
            placeholder="Please select your rating of the food quality..."
        )

        s = st.selectbox(
            "What is your rating on the service? ",
            options = ['Just okay', 'Could be better', 'Meets expectations', 'Very friendly', 'Super Kind'], 
            index=None, 
            placeholder="Please select your rating of the food quality..."
        )

        u = st.selectbox(
            "What is your rating on the unique aspects? This should reflect your opinion on the decor, stand out features, unique offers, etc.",
            options = ['Not my favorite', 'Mediocre', 'Decent', 'Very good', 'Outstanding'], 
            index=None, 
            placeholder="Please select your rating of unique aspects..."
        )

        positive_review = st.text_area('Please give a positive review about the location:',
                                       placeholder='Please fill in your positive review...')
        
        negative_review = st.text_area("Please give a negative review about the location:",
                                       placeholder="Please fill in your negative review...")
        would_go_back = st.selectbox(
            "Would you go back?",
            options=["Definitely Not", "Probably Not", "Maybe", "Probably Yes", "Absolutely"],
            index=None,
            placeholder="Please select an option...")

        st.subheader('Accessbility')

        parking_ease = st.selectbox(
            "What was your experience getting to the location?",
            options = ['Parked 20 blocks away', 'Kept circling to find a spot', 'It was manageable', 'Plenty of spaces close by', 'So easy and stress-free'], 
            index=None, 
            placeholder="Please select an option that best reflects your experience..."
        )

        parkingtype1 = st.selectbox(
            "What was your primary parking type?",
            options = ['Street', 'Parking Lot', 'Parking Garage', 'Drive Through', 'Uber Eats', 'Door Dash', 'Metro'],
            index=None, 
            placeholder="Please select your primary parking type..."
        )

        parkingtype2 = st.selectbox(
            "What was your secondary parking type?",
            options = ['Street', 'Parking Lot', 'Parking Garage', 'Drive Through', 'Uber Eats', 'Door Dash', 'Metro', 'None'],
            index=None, 
            placeholder="Please select your secondary parking type..."
        )

        st.subheader("Techical Compatibilty")

        wifi_select = st.selectbox(
            'Did the location offer free wifi?',
            options=['Yes', 'No', 'Not Sure'],
            index=None,
            placeholder="Please select an option..."
        )

        charging_select = st.selectbox(
            'Did the location offer charging outlets?',
            options=['Yes', 'No', 'Not Sure'],
            index=None,
            placeholder="Please select an option..."
        )

        valid_input = all([name, address, category, season, 
                       order, p, r, p, a, f,
                         s,u, positive_review,negative_review,
                         would_go_back,parking_ease,parkingtype1,
                           parkingtype2,wifi_select, charging_select])
        
        submit_button = st.form_submit_button('Submit', disabled=not valid_input, type='primary')

    if submit_button: 
        time = datetime.datetime.now(tz='EST').strftime("%m/%d/%Y")
        new_data = pd.DataFrame({
            "Timestamp" : [time], 
            "Name" : [name],
            "Location": [address],
            "Category": [category],
            "Season Visited": [season], 
            "What I got/did" : [order],
            "Rating": [r], 
            "Price": [p],
            "Atmosphere": [a], 
            "Food quality": [f],
            "Service":[s],
            "Unique Aspects": [u], 
            "Positive Review": [positive_review], 
            "Negative Review": [negative_review],
            "Would go back?": [would_go_back], 
            "Parking": [parking_ease],
            "WiFi": [wifi_select], 
            "Charging Outlets": [charging_select]
        })

        try: 
            existing_data = pd.read_csv(csv_file)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        except FileNotFoundError: 
            updated_data=new_data

        updated_data.to_csv(csv_file, index=False)
        st.success(f"Thank you for contributing to this project! Your review was submitted on {time}", icon="✅")

        
elif page == "Data at a Glance":
    st.title("Eats & Adventures Tracker - Data at a Glance")
    # Map Visualization 

    st.markdown("""
    <style>
    span[data-baseweb="tag"] {
    color: black;
    background-color: white
    }
    <style>
    """, unsafe_allow_html=True)

    selected_category = st.multiselect(
        label="Select a Category", 
        options=df['Category'].unique(),
        placeholder="Select a Category"
    )

    if selected_category:
        filtered_states = df[df['Category'].isin(selected_category)]['State'].unique()
    else: 
        filtered_states = []

    selected_state = st.multiselect(
        label="Select a State/Territory", 
        options=filtered_states,
        placeholder="Select a State"
    )

    if selected_state:
        filtered_cities = df[df['Category'].isin(selected_category) & df['State'].isin(selected_state)]['City'].unique()
    else: 
        filtered_cities = []

    selected_city = st.multiselect(
            label="Select a City/Territory", 
            options= filtered_cities,
            placeholder="Select a City"
        )

    st.header("Map of Reviewed Locations")

    if selected_state or selected_city or selected_category:

        filtered_df = df

        if selected_state: filtered_df = filtered_df[(filtered_df['State'].isin(selected_state))]
        if selected_city: filtered_df = filtered_df[(filtered_df['City'].isin(selected_city))]
        if selected_category: filtered_df = filtered_df[(filtered_df['Category'].isin(selected_category))]

        map_fig = px.scatter_mapbox(
            filtered_df,
            lat='lat',
            lon='lon',
            size='size',
            hover_name='Name',
            color='Rating',  # Use the Rating column for color mapping
            color_continuous_scale='Purpor',
            mapbox_style='carto-positron',
            title='Reviewed Locations:',
            width=1000,
            height=700,
            zoom=5
        )

        map_fig.update_traces(
        hovertemplate="<b>%{hovertext}</b>"  # Only show the name
        )

        st.plotly_chart(map_fig)

    else:
        full_map = px.scatter_mapbox(
            df,
            lat='lat',
            lon='lon',
            size='size',
            hover_name='Name',
            color='Rating',  # Use the Rating column for color mapping
            color_continuous_scale='Purpor',
            mapbox_style='carto-positron',
            title='Reviewed Locations:',
            width=1000,
            height=700, 
            zoom = 5
        )
        full_map.update_traces(
        hovertemplate="<b>%{hovertext}</b>"  # Only show the name
        )
        st.plotly_chart(full_map)

    # Reviews Over Time

    st.header("Reviews Over Time")

    df_group = df.groupby('Timestamp')['Count of Reviews'].sum().reset_index()
    df_group2 = df.groupby('Timestamp')['Rating'].median().reset_index()
    df_group['Median Rating'] = df_group2['Rating']

    time_bar = px.bar(
        df_group, 
        x = 'Timestamp',
        y = 'Count of Reviews', 
        color = 'Median Rating', 
        color_continuous_scale= 'Purpor', 
        width=1000, 
        height = 300, 
        hover_name='Count of Reviews'
    )  

    time_bar.update_traces(
    hovertemplate="<b>%{hovertext}</b>"  
    )

    time_bar.update_layout(
    title="Count of Reviews Over Time",
    xaxis_title="Date",
    yaxis_title="Count of Reviews"
    )

    st.plotly_chart(time_bar)

    st.markdown(
        """
        Note: You may notice a high count of reviews in the begining of the bar chart around 11/11 and 11/12. After collecting using a notesheet over the summer, I transitioned into a more uniformed process starting in Novemember, hence the large spike in the bar plot. 
        """
    )

    st.markdown(
        'The following pie chart shows the percentage of categories represented in the reviews.'
    )
    
    pie_col1, pie_col2 = st.columns(2)

    with pie_col1: 
        df_pie = df.groupby('Category')['Count of Reviews'].sum().reset_index()
        df_pie['Count of Reviews'] = df_pie['Count of Reviews']/100
        df_pie = df_pie.sort_values(by='Category', ascending=True)
        pie_chart = px.pie(df_pie, names = 'Category', values='Count of Reviews', 
                            title='Percentage of Categories Represented in Reviews',
                            color_discrete_sequence=px.colors.sequential.Sunsetdark, 
    )
        st.plotly_chart(pie_chart)

    with pie_col2: 
        cat_rating_bar = df.groupby('Category')['Rating'].median().reset_index()
        cat_rating_bar = cat_rating_bar.sort_values(by='Rating', ascending=True)
        bar_chart = px.bar(cat_rating_bar, x='Category', y='Rating', title='Median Rating Across Categories', color_discrete_sequence=px.colors.sequential.PuBu)
        st.plotly_chart(bar_chart)

    pie_col3, pie_col4 = st.columns(2)

    with pie_col3: 

        pie_chart3 = df.groupby('Parking Type 1')['Count of Reviews'].sum().reset_index()
        pie_chart3['Count of Reviews'] = pie_chart3['Count of Reviews']/100
        pie_chart3 = pie_chart3.sort_values(by='Parking Type 1', ascending=False)
        pie_chart3 = px.pie(pie_chart3, names = 'Parking Type 1', values='Count of Reviews', 
                            title='Primary Parking Type Represented in Dataset', 
                            color_discrete_sequence=px.colors.sequential.Purples[2:]
    )
        st.plotly_chart(pie_chart3)



    with pie_col4: 
        pie_chart4 = df.groupby('Parking Type 2')['Count of Reviews'].sum().reset_index()
        pie_chart4['Count of Reviews'] = pie_chart4['Count of Reviews']/100
        pie_chart4 = pie_chart4.sort_values(by='Parking Type 2', ascending=False)
        pie_chart4 = px.pie(pie_chart4, names = 'Parking Type 2', values='Count of Reviews', 
                            title='Secondary Parking Type Represented in Dataset',
                            color_discrete_sequence=px.colors.sequential.Purpor[:10]

    )
        st.plotly_chart(pie_chart4)

elif page == "My Recommendations": 
    st.title("Eats & Adventures Tracker - My Recommendations")
    
    recommendations(df)
    
elif page == "Your Recommendations": 

    st.title("Eats & Adventures Tracker - Your Recommendations")

    #TODO: data frame formatting
    df2 = pd.read_csv("form_submission.csv")

    df2['Count of Reviews'] = 1
    df2['Overall Review'] = df2['Positive Review'] + " " + df2['Negative Review']
    df2['color'] = df2['Rating'].apply(get_color)
    df2['size'] = df2['Rating'].apply(get_size)
    df2['Rating'] = df2['Rating'].to_list()
    df2['Overall Sentiment'] = df2['Overall Review'].apply(clean).apply(get_overall_sentiment)
    df2['Atmosphere'] = df2['Atmosphere'].apply(atmosphere)
    df2['Rating'] = df2['Rating'].apply(rating)
    df2['Food quality'] = df2['Food quality'].apply(food_quality)
    df2['Service'] = df2['Service'].apply(service)
    df2['Unique Aspects'] = df2['Unique Aspects'].apply(unique_aspects)
    df2['Would go back?'] = df2['Would go back?'].apply(go_back)
    df2['Latitude'] = df2['Location'].apply(lat)
    df2['Longitude'] = df2['Location'].apply(lon)
    df2['City'] = df2['Location'].apply(city)
    df2['State'] = df2['Location'].apply(state)
    df2['Price Range'] = df2['Price']

    st.dataframe(df2)
    recommendations(df=df2)

    #TODO: Add a form entry here and a progress bar for how much of the form is complete in the second column

elif page == "Give me feedback": 
    st.title("Eats & Adventures Tracker - Give me feedback")

