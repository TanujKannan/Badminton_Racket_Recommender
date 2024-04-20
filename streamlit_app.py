import streamlit as st
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the trained model and LabelEncoder
model = joblib.load('racket_prediction_model.joblib')
le = joblib.load('label_encoder.joblib')

# Load the dataset
df = pd.read_csv('badminton_dataset.csv')

# Streamlit UI
st.title('🏸 Badminton Racquet Recommender')
st.markdown('Welcome to the Badminton Racquet Recommender! This tool helps you find the perfect racquet based on your preferences.')

# Input fields
st.sidebar.title('Preferences')
param1 = st.sidebar.slider('Skill Level', min_value=1, max_value=10, value=5, step=1)
param2 = st.sidebar.slider('Attack Potential', min_value=1, max_value=10, value=5, step=1)
param3 = st.sidebar.slider('Defense Potential', min_value=1, max_value=10, value=5, step=1)
param4 = st.sidebar.slider('Budget', min_value=1, max_value=10, value=5, step=1)
param5 = st.sidebar.slider('Flexibility', min_value=1, max_value=10, value=5, step=1)

st.sidebar.markdown('Adjust the sliders according to your preferences.')

# Initialize submit_feedback as False
submit_feedback = False

# Checkbox for user feedback
feedback_checkbox = st.sidebar.checkbox('Provide feedback by clicking the checkbox!')

# Dropdown for selecting preferred racket
if feedback_checkbox:
    unique_rackets = df['Racket_Name'].drop_duplicates().tolist()
    preferred_racket = st.sidebar.selectbox('Select your preferred racquet', unique_rackets)
    submit_feedback = st.sidebar.button('Submit Feedback')

# Button to make prediction
if st.sidebar.button('Get Recommended Racquet', key='predict_button'):
    # Make prediction
    input_features = [[param1, param2, param3, param4, param5]]
    encoded_prediction = model.predict(input_features)[0]

    # Decode the prediction using the LabelEncoder
    decoded_prediction = le.inverse_transform([encoded_prediction])[0]

    # Display the predicted racket name
    st.subheader('Recommended Badminton Racquet:')
    st.success(f'🏆 {decoded_prediction} 🏆')
    st.markdown('Congratulations! You\'ve got the perfect racquet for your game.')

# Add user feedback to the dataset and retrain the model
if submit_feedback:
    new_entry = pd.DataFrame([{'Skill': param1, 'Attack_Potential': param2, 'Defense_Potential': param3,
                                'Racket_Price': param4, 'Racket_Flexibility': param5, 'Racket_Name': preferred_racket}])
    df = pd.concat([df, new_entry], ignore_index=True)
    # Separate features and target variable from the updated dataset
    X_updated = df.drop('Racket_Name', axis=1)
    y_updated = df['Racket_Name']
    # Encode the categorical target variable (Racket_Name) using the same LabelEncoder
    y_updated = le.transform(y_updated)
    # Retrain the existing model with the updated dataset
    model.fit(X_updated, y_updated)
    # Save the updated model
    joblib.dump(model, 'racket_prediction_model.joblib')
    st.success('Your feedback submitted successfully! This helps improve our model! Please change the sliders to check out other recommendations.')
    feedback_checkbox = False
