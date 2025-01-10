import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib  # Assuming you save the model with joblib

# Load the trained model and preprocessing components
rfc = joblib.load('model.pkl')  # Load your saved model
la = joblib.load('label_encoder.pkl')  # Load the saved scaler if you saved it during training
sc=joblib.load('scaler.pkl')
# Streamlit UI
st.title("hi")
st.write("Ee")