import streamlit as st
from numpy import load
from pandas import read_csv
from PIL import Image
from utils import *

# Page logo and necessary informations
image = Image.open('diabetes.jpg')
st.image(image, use_column_width=True)
st.title('Diabetes Prediction - Group 06')
st.write('Select the patient ID you want to check and what you want to see')

# Load patients list
patients = load('../pipeline/Patients_info/patients.npy')

# Creation of Selectbox and Radio buttons
left, right = st.columns(2)
with left:
    st.markdown("<div style='height: 1.4rem'></div>", unsafe_allow_html=True)
    patient = st.selectbox(' ', patients, label_visibility='hidden')
with right:
    st.markdown("<div id='wewe1' style='height: 0rem'></div>", unsafe_allow_html=True)
    st.session_state['plot'] = st.radio(' ', ['History', 'Prediction'], label_visibility='hidden')

st.markdown("<div style='height: 3.4rem'></div>", unsafe_allow_html=True)

# Case of History visualization
if st.session_state['plot'] == 'History':
    # Plot patient history
    image = Image.open(f'../pipeline/Patients_info/{patient}/{patient}_data_history.png')
    st.image(image, use_column_width=True)
# Case of Prediction visualization
elif st.session_state['plot'] == 'Prediction':
    # Calculate actual, minimum and maximum glucose values
    raw_dataset = read_csv(f'../pipeline/Datasets/{patient}/{patient}-ws-testing(t+30).csv')
    test_values = raw_dataset.values[-1000:, -1]
    
    left, center, right = st.columns(3)
    with left:
        st.metric(label='Current Glucose Level', value=f'{round(test_values[-1], 2)} mg/dL')
    with center:
        st.metric(label='Minimum Glucose Level', value=f'{round(min(test_values), 2)} mg/dL')
    with right:
        st.metric(label='Maximum Glucose Level', value=f'{round(max(test_values), 2)} mg/dL')

    st.markdown('''---''')
    st.markdown("<div style='height: 1.4rem'></div>", unsafe_allow_html=True)

    # Load test_X and test_y and model calculated in the pipeline
    test_X = load(f'../pipeline/Patients_info/{patient}/{patient}_test_X.npy')
    test_y = load(f'../pipeline/Patients_info/{patient}/{patient}_test_y.npy')
    model = keras.models.load_model(f'../pipeline/Patients_info/{patient}/{patient}_model.h5')
    
    # Create the model of a new patient and make the prediction
    real, predicted = test_model(test_X, test_y, model, patient)
    
    left, right = st.columns(2)
    with left:
        # Plot patient prediction
        image = Image.open(f'../pipeline/Patients_info/{patient}/{patient}_real_pred.png')
        st.image(image, use_column_width=True)
    with right:
        # Plot patient clarke error grid
        image = Image.open(f'../pipeline/Patients_info/{patient}/{patient}_clarke.png')
        st.image(image, use_column_width=True)