import streamlit as st
import joblib

# Load the trained model and LabelEncoder
model = joblib.load('racket_prediction_model.joblib')
le = joblib.load('label_encoder.joblib')

# Streamlit UI
st.title('Badminton Racket Predictor')
st.subheader('A higher number for price, indicates a higher budget')

# Input fields
param1 = st.slider('Skill', min_value=1, max_value=10)
param2 = st.slider('Attack Potential', min_value=1, max_value=10)
param3 = st.slider('Defense Potential', min_value=1, max_value=10)
param4 = st.slider('Racket Price', min_value=1, max_value=10)
param5 = st.slider('Racket Flexibility', min_value=1, max_value=10)

# Button to make prediction
if st.button('Get Recommended Racket'):
    # Make prediction
    input_features = [[param1, param2, param3, param4, param5]]
    encoded_prediction = model.predict(input_features)[0]

    # Decode the prediction using the LabelEncoder
    decoded_prediction = le.inverse_transform([encoded_prediction])[0]

    # Display the predicted racket name
    st.subheader('Recommended Badminton Racket:')
    st.write(decoded_prediction)
