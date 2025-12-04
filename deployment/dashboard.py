import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="ML Dashboard", page_icon="🤖")

# Load model and scaler
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("dataprocessed/scaler.pkl")

# Get actual feature names from training data
@st.cache_data
def get_feature_names():
    train_df = pd.read_csv("dataprocessed/train.csv")
    return [col for col in train_df.columns if col != 'target']

st.title("🤖 ML Model Dashboard")

page = st.sidebar.radio("Navigation", ["Home", "Make Prediction", "Model Info"])

if page == "Home":
    st.header("Welcome!")
    st.write("Use the sidebar to navigate.")
    st.metric("Model Status", "✅ Active")

elif page == "Make Prediction":
    st.header("Make a Prediction")
    
    model = load_model()
    scaler = load_scaler()
    feature_names = get_feature_names()
    
    # Create input fields for key features
    st.subheader("Enter Feature Values")
    
    # Create a dictionary to store all features with default values
    features = {}
    
    # Display only the first 10 features for simplicity
    st.write("**Main Features:**")
    col1, col2 = st.columns(2)
    
    for i, feature_name in enumerate(feature_names[:10]):
        if i % 2 == 0:
            with col1:
                features[feature_name] = st.number_input(
                    feature_name, 
                    value=0.0,
                    format="%.4f"
                )
        else:
            with col2:
                features[feature_name] = st.number_input(
                    feature_name, 
                    value=0.0,
                    format="%.4f"
                )
    
    # Set remaining features to 0
    for feature_name in feature_names[10:]:
        features[feature_name] = 0.0
    
    if st.button("Predict"):
        # Create DataFrame with correct feature names
        df = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0].max()
        
        st.success(f"Prediction: {'Malignant' if prediction == 1 else 'Benign'}")
        st.info(f"Confidence: {probability:.2%}")
        
        # Show prediction probabilities
        proba = model.predict_proba(df)[0]
        st.write("**Class Probabilities:**")
        st.write(f"- Benign (0): {proba[0]:.2%}")
        st.write(f"- Malignant (1): {proba[1]:.2%}")

elif page == "Model Info":
    st.header("Model Information")
    model = load_model()
    feature_names = get_feature_names()
    
    st.write(f"**Model Type:** {type(model).__name__}")
    st.write(f"**Number of Features:** {model.n_features_in_}")
    
    st.subheader("Feature Names:")
    for i, name in enumerate(feature_names, 1):
        st.write(f"{i}. {name}")