import streamlit as st
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model, label encoder, and tokenizer
MODEL_PATH = 'logistic_regression_model.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
TOKENIZER_PATH = 'tokenizer.pkl'

# Load your models and other necessary files
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
        logging.info("Model loaded successfully.")

    with open(LABEL_ENCODER_PATH, 'rb') as file:
        label_encoder = pickle.load(file)
        logging.info("Label encoder loaded successfully.")

    with open(TOKENIZER_PATH, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
        logging.info("TF-IDF vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model files: {e}")

# Streamlit web app
st.title("BBC News Classifier")

# Text input
text = st.text_area("Enter text to classify:")

# Prediction button
if st.button("Classify Text"):
    if not text:
        st.error("Please enter some text.")
    else:
        try:
            # Transform the text input
            tfidf_features = tfidf_vectorizer.transform([text])
            # Predict the category
            prediction = model.predict(tfidf_features)
            # Convert the numeric label to the category name
            category = label_encoder.inverse_transform(prediction)[0]
            st.success(f"Predicted Category: {category}")
        except Exception as e:
            st.error("An error occurred during prediction.")
            logging.error(f"Prediction error: {e}")
