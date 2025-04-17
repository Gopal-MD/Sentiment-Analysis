import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.emotion_utils import extract_emotion_sentiment

# Load custom CSS for styling
try:
    with open("styles/custom.css") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error("Custom CSS file not found! Please ensure 'styles/custom.css' exists.")

# Streamlit app title and description
st.title("Decoding Emotions: Sentiment Analysis of Social Media Conversations")
st.write("Analyze emotions and sentiments expressed in social media text data.")

# File upload functionality
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

if uploaded_file:
    # Read uploaded CSV
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(data)
    except Exception as e:
        st.error(f"Error reading the uploaded file: {e}")
        st.stop()

    # Ensure the 'text' column exists
    if 'text' not in data.columns:
        st.error("The uploaded file must contain a 'text' column.")
    else:
        # Apply sentiment analysis with a progress spinner
        st.write("Analyzing emotions and sentiments...")
        with st.spinner("Processing data..."):
            try:
                data['emotion'], data['sentiment'] = zip(*data['text'].apply(extract_emotion_sentiment))
                st.success("Analysis completed successfully!")
            except Exception as e:
                st.error(f"Error during sentiment analysis: {e}")
                st.stop()

        # Display results
        st.write("Analysis Results:")
        st.dataframe(data)

        # Download results
        st.download_button(
            label="Download Results as CSV",
            data=data.to_csv(index=False),
            file_name="results.csv",
            mime="text/csv"
        )

        # Visualize Emotion Distribution
        st.write("Emotion Distribution:")
        fig, ax = plt.subplots()
        try:
            sns.countplot(x='emotion', data=data, ax=ax, palette="pastel")
            ax.set_title("Emotion Distribution")
            ax.set_xlabel("Emotion")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error visualizing emotion distribution: {e}")

        # Visualize Sentiment Distribution
        st.write("Sentiment Distribution:")
        fig, ax = plt.subplots()
        try:
            sns.countplot(x='sentiment', data=data, ax=ax, palette="muted")
            ax.set_title("Sentiment Distribution")
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error visualizing sentiment distribution: {e}")