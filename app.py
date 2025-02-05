import streamlit as st
import pickle

# Load the saved sentiment model
with open("sentiment_model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

# Load the saved TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit UI
st.title("📊 Customer Reviews Sentiment Analysis")

st.write("### Enter a review to analyze sentiment:")

# User Input
user_input = st.text_area("Type your review here...")

if st.button("Analyze Sentiment"):
    input_vectorized = vectorizer.transform([user_input])  # Transform input using the vectorizer
    prediction = classifier.predict(input_vectorized)[0]  # Predict sentiment

    # Display Result
    st.write(f"**Predicted Sentiment: {prediction}**")
