import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
# Load the pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

#clean the text
def clean_text(text):
  text = re.sub(r"[^a-zA-Z]"," ",text).lower()
  #the above line will replace all expressions other than aplhabets to spaces, and to make machine to understand all the letters are converted into lower as machine is case sensitive-
  #machine asumes AS and as as 2 different words
  tokens = text.split()
  tokens= [word for word in tokens if word not in stopwords]
  return " ".join(tokens)

# Streamlit app
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("Sentiment Analysis App")
st.write("Enter a movie review to analyze its sentiment (positive or negative).")

# User input
user_input = st.text_area("Enter your review:")

if st.button("Analyze sentiment"):
    # Preprocess the input
    cleaned_input = clean_text(user_input)
    vectorized_input = vectorizer.transform([cleaned_input])

    # Make prediction
    prediction = model.predict(vectorized_input)

    # Display result
    if prediction[0] == 1:
        st.success("Positive sentiment detected.")
    else:
        st.error("Negative sentiment detected.")