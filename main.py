from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd

def predict_charge(model, df):
    predictions = predict_model(estimator=model, data=df)
    return int(predictions['Label'][0])

model = load_model('gbr_insurance')

st.title('Annual Medical Charges Prediction App')
st.write(
    ''' 
        This is a web app to predict the Annual Medical Charges 
        of new customers based on their information.
    '''
)

age = st.sidebar.slider(
    label='Age', 
    min_value=18.0,
    max_value=64.0,
    step=1.0
)

sex = st.sidebar.selectbox(
    'What is your sex',
    ('male','female')
)

bmi = st.sidebar.slider(
    label='BMI (Body Mass Index',
    min_value = 15.0,
    max_value = 54.0,
    step=0.1
)

children = st.sidebar.selectbox(
    'How many children do you have?',
    (0, 1, 2, 3, 4, 5)
)

smoker = st.sidebar.selectbox(
    'Do you smoke?',
    ('yes', 'no')
)

region = st.sidebar.selectbox(
    'Region',
    ('southwest', 'southeast', 'northwest', 'northeast')
)

features = {
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'children': children,
    'smoker': smoker,
    'region': region,
}

features_df = pd.DataFrame([features])

st.table(features_df)

if st.button('Predict'):
    prediction = predict_charge(model, features_df)
    
    st.write(' Based on feature values, your Annual Medical Charge is $' + str(prediction))