import streamlit as st
import numpy 
import pandas
import pickle
import re
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
model = tf.keras.models.load_model(r'D:\sentiment_analysis_model.h5')
tokenizer_path = r'D:\tokenizer.pkl'

with open(tokenizer_path,'rb') as file:
    tokenizer = pickle.load(file)

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'\d+',' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(f"[{string.punctuation}]",' ',text)
    return text

def predict_sentiment(input):
    text = tokenizer.texts_to_sequences([clean_text(input)])
    pad_text =pad_sequences(text,maxlen=166,padding='post')
    prediction = model.predict(pad_text)
    return prediction[0][0]

st.title('Sentiment Analysis App')

user_input = st.text_area("Enter text for Sentiment Analysis")

if st.button('Predict'):
    if user_input:
        score = predict_sentiment(user_input)
        st.write(f'Sentiment score :{score:.2f}')
        sentiment = 'Positive' if score > 0.5 else 'Negative'
        st.write(f'Sentiment :{sentiment}')
    else:
        st.write('Enter text to analyze')


