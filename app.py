import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and feature names
@st.cache_resource
def load_model():
    model = joblib.load('misinformation_model.pkl')
    features = joblib.load('feature_names.pkl')
    return model, features

model, features = load_model()

# App Title
st.title("🤞AI Misinformation Detector🤞")
st.markdown("Enter post characteristics to predict if it is **Misinformation**.")

# Sidebar for inputs
st.sidebar.header("Post Metrics")    

def user_input_features():
    # Create input fields for each feature
    data = {}
    data['author_followers'] = st.sidebar.number_input("Author Followers", min_value=0, value=1000)
    data['author_verified'] = st.sidebar.selectbox("Author Verified", [0, 1])
    data['text_length'] = st.sidebar.slider("Text Length", 10, 500, 150)
    data['token_count'] = st.sidebar.slider("Token Count", 1, 100, 30)
    data['readability_score'] = st.sidebar.slider("Readability Score", 0.0, 100.0, 50.0)
    data['num_urls'] = st.sidebar.number_input("Number of URLs", 0, 10, 1)
    data['num_mentions'] = st.sidebar.number_input("Number of Mentions", 0, 10, 0)
    data['num_hashtags'] = st.sidebar.number_input("Number of Hashtags", 0, 10, 0)
    data['sentiment_score'] = st.sidebar.slider("Sentiment Score (-1 to 1)", -1.0, 1.0, 0.0)
    data['toxicity_score'] = st.sidebar.slider("Toxicity Score (0 to 1)", 0.0, 1.0, 0.2)
    data['detected_synthetic_score'] = st.sidebar.slider("Synthetic/AI Score", 0.0, 1.0, 0.5)
    data['embedding_sim_to_facts'] = st.sidebar.slider("Fact Similarity Score", 0.0, 1.0, 0.5)
    data['external_factchecks_count'] = st.sidebar.number_input("External Fact Checks", 0, 10, 0)
    data['source_domain_reliability'] = st.sidebar.slider("Source Reliability (0 to 1)", 0.0, 1.0, 0.8)
    data['engagement'] = st.sidebar.number_input("Engagement (Likes/Shares)", 0, 10000, 500)
    
    return pd.DataFrame([data])

# Get user input
input_df = user_input_features()

# Display Input parameters
st.subheader("Input Post Parameters")
st.write(input_df)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ This post is likely **MISINFORMATION**.")
    else:
        st.success("✅ This post is likely **NOT MISINFORMATION**.")

    st.subheader("Prediction Probability")
    st.write(f"Misinformation: {prediction_proba[0][1]*100:.2f}%")
    st.write(f"Authentic: {prediction_proba[0][0]*100:.2f}%")