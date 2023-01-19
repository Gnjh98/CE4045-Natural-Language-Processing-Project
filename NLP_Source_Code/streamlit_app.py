import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud  

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://media.idownloadblog.com/wp-content/uploads/2022/08/Apple-Far-Out-Event-Wallpaper-for-6K-desktop.png");
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

#############
# [BACKEND] #
#############
# Data Pre-processing (extracted)
def removePunctuations(tweet):
    return re.sub(r'[^\w\s]', '', tweet)
def removeNumbers(tweet):
    return re.sub(r'[0-9]', '', tweet)
def dataPreprocessing(tweet):
    tweet = removePunctuations(tweet)
    tweet = removeNumbers(tweet)
    return tweet.lower().strip()
def tokenize(text):
    return word_tokenize(text)
def remove_stopword(text):
    nltk_tokenList = tokenize(text)
    filtered_sentence = []
    nltk_stop_words = set(stopwords.words("english"))
    for w in nltk_tokenList:  
        if w not in nltk_stop_words:  
            filtered_sentence.append(w)
    return ' '.join(filtered_sentence)


# Fixed list of categories about Apple event
list_category = ["appleevent","iphone","watch"]

# Find category of tweet
def find_category_keyword(list_text, keyword):
    for i in list_text:
        if keyword in i:
            return True     
    return False

# Remove categorical keyword from tweet
def remove_category_keyword(list_text):
    i = 0
    while i < len(list_text):
        for j in list_category:
            if j in list_text[i]:
                list_text.pop(i)
                if len(list_text) == 0:
                    break
                i -= 1
        i += 1   
    return ' '.join(list_text) # join back tweet

# Retrieve sentiment data
overall_url = "https://raw.githubusercontent.com/X-Yang98/CE4045-NLP/main/combined_dataset.csv"
df = pd.read_csv(overall_url)
df['no_stopword_tweet'] = df['tweet'].apply(remove_stopword)
df['tokenize'] = df['no_stopword_tweet'].apply(tokenize)
for category in list_category:
    df[category] = df['tokenize'].apply(find_category_keyword, args=(category,))
df['no_category_tweet'] = df['tokenize'].apply(remove_category_keyword)

# WordCloud generator function
def generate_wordcloud(df):
    string = pd.Series(df).str.cat(sep=' ')

    wordcloud = WordCloud(width=1600, height=800, 
                      max_font_size=200, max_words=50, collocations=False, 
                      background_color='black').generate(string)

    fig = plt.figure(figsize=(40,30))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return fig
  

#############
#[FRONTEND] #
#############
st.title("Twitter Sentiment Analysis on #AppleEvent2022")

# Loading model and vectorizer into streamlit
load_model_state = st.text("Loading trained classifier...")
vectorizer = pickle.load(open("/content/vectorizer.pkl", "rb"))
model = pickle.load(open("/content/logistic_classifier.pkl", "rb"))
load_model_state.text("Trained classifier loaded!")

# Upload training csv/txt file
st.subheader("Upload test file to classify")
with st.form("placeholder", clear_on_submit = True):
    uploaded_file = st.file_uploader("Choose test file")
    uploaded_button = st.form_submit_button("Upload and classify")
    if uploaded_button and uploaded_file is not None:
        data = uploaded_file.getvalue().decode('utf-8').splitlines()
        headers = data[0].split(sep = ",")
        data_dict = {header : [] for header in headers}
        for i in range(1, len(data)):
            split_data = data[i].split(sep = ",")
            for e in range(len(headers)):
                data_dict[headers[e]].append(split_data[e])
        test_df = pd.DataFrame.from_dict(data_dict)

        test_df = test_df.drop_duplicates()
        test_df = test_df.dropna()
        st.text("Unclassified test dataframe:")
        st.write(test_df)

        classify_state = st.text("Classifying test data...")
        x_test = vectorizer.transform(test_df[headers[0]])
        y_pred = model.predict(x_test)
        classified_df = test_df.copy()
        classified_df["predicted_labels"] = y_pred
        classify_state.text("Displaying classified test data")
        st.text("Classified test dataframe:")
        st.write(classified_df)
        st.write("Our model predicted with an accuracy of {}%.".format(round(accuracy_score(test_df[headers[1]], y_pred), 2)))

    elif uploaded_button:
        st.write("Please upload a valid csv file with columns (tweets, label)")

# Taking in a query sentence as input
raw_sentence = st.text_input(label="Ask me about #AppleEvent2022!", placeholder="And I will give you a surprise!")
raw_category = st.selectbox('Which hashtag?', ('No Hashtag', '#AppleEvent2022', '#iPhone14'))
state = st.button("Find out")
st.write("*our dataset consist of tweets scraped on 26 September 2022")

# Analyse Sentiment of input by user
if state:
    if not raw_category: # check if 1 of hashtags is selected
        st.write("Select a hashtag!")
    else:
        # Process input
        sentence = dataPreprocessing(raw_sentence)

        # Model prediction on sentence sentiment
        y_pred = model.predict(vectorizer.transform([sentence])) 
        sentiment = y_pred[0]

        # Find category
        category = ''
        if raw_category == 'No Hashtag':
            category = removeNumbers(raw_category[1:].lower()) # lowercase + remove numbers

        # Filter dataset for similar sentiments base on category
        sentiment_df = df.loc[df['final_label'] == sentiment]
        if len(category) > 0:
            user_df = sentiment_df.loc[sentiment_df[category]]
            word_cloud = generate_wordcloud(user_df['no_category_tweet'])
        else:
            word_cloud = generate_wordcloud(sentiment_df['no_category_tweet'])

        word_cloud_state = st.text("Generating word cloud on similar sentiments...")
        st.pyplot(word_cloud)
        word_cloud_state.text("Your sentiment is {}.".format(sentiment.upper()))
        st.write("{}% of people on Twitter in our dataset* thinks like you.".format(round(len(sentiment_df)/len(df)*100), 2))
