import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the trained model and preprocessing components
rfc = joblib.load('model.pkl')  # Load your saved RandomForestRegressor model
la = joblib.load('label_encoder.pkl')  # Load your saved LabelEncoder
sc = joblib.load('scaler.pkl')  # Load your saved StandardScaler

# Streamlit UI
st.title("Shoe Price Prediction App")
st.subheader("Predict the offer price of shoes based on various attributes")

# User input for features
st.sidebar.header("Input Features")
brand = st.sidebar.selectbox("Select Brand", ['Tresmode', 'Lavie', 'FILA', 'Crocs'])  # Example brands
color = st.sidebar.selectbox("Select Color", ['Black', 'Gold', 'Red', 'Blue'])  # Example colors
size = st.sidebar.number_input("Enter Shoe Size", min_value=4.0, max_value=12.0, step=0.5)
price = st.sidebar.number_input("Enter Original Price", min_value=100.0, step=50.0)

# Handle unseen labels for the brand
if brand not in la.classes_:
    la.classes_ = np.append(la.classes_, brand)

# Preprocess user inputs
brand_encoded = la.transform([brand])  # Encode brand
color_encoded = la.transform([color])  # Encode color

# Combine all features into a single array
input_features = np.array([brand_encoded[0], color_encoded[0], size, price])

# Scale all features (brand, color, size, price)
scaled_features = sc.transform([input_features])

# Make prediction
if st.button("Predict Offer Price"):
    prediction = rfc.predict(scaled_features)[0]
    st.write(f"Predicted Offer Price: ₹{prediction:.2f}")

# Visualization
st.header("Visualize Data Insights")
if st.checkbox("Show Histograms"):
    st.write("Displaying histograms for the dataset")
    # Add histograms or visuals using sample data or loaded data

if st.checkbox("Show Boxplot"):
    st.write("Displaying boxplots for price and offer price")
    # Add boxplots or any specific visualizations
