import streamlit as st
import joblib
import numpy as np
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

st.title("Income Prediction App")
st.write("Predict whether income exceeds $50K/yr")

# --- Inputs ---
age = st.number_input("Age", min_value=17, max_value=90, value=30)

workclass = st.selectbox("Workclass", [
    'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
    'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
])

fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=0, value=100000)

education = st.selectbox("Education", [
    'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th-10th',
    '11th-12th', 'HS-grad', 'Some-college', 'Assoc-voc',
    'Assoc-acdm', 'Bachelors', 'Post-Grad'
])

education_num = st.number_input("Education Number", min_value=1, max_value=16, value=10)

marital_status = st.selectbox("Marital Status", [
    'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
    'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
])

occupation = st.selectbox("Occupation", [
    'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
    'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
    'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
    'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
])

relationship = st.selectbox("Relationship", [
    'Wife', 'Own-child', 'Husband', 'Not-in-family',
    'Other-relative', 'Unmarried'
])

race = st.selectbox("Race", [
    'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
    'Other', 'Black'
])

sex = st.selectbox("Sex", ['Male', 'Female'])

capital_gain = st.number_input("Capital Gain", min_value=0, value=0)

capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)

native_country = st.selectbox("Native Country", [
    'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico', 'South',
    'Japan', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada',
    'El-Salvador', 'England', 'China', 'Taiwan', 'Iran', 'Haiti',
    'Portugal', 'Vietnam', 'Italy', 'Poland', 'Columbia', 'Cambodia',
    'Thailand', 'Ecuador', 'Laos', 'Yugoslavia', 'Peru',
    'Dominican-Republic', 'Guatemala', 'Scotland', 'Honduras',
    'Hungary', 'Trinadad&Tobago', 'Nicaragua', 'Greece', 'France',
    'Ireland', 'Hong', 'Outlying-US(Guam-USVI-etc)'
])

#Predict
if st.button("Predict"):
    input_data = pd.DataFrame([[
        age, workclass, fnlwgt, education, education_num,
        marital_status, occupation, relationship, race, sex,
        capital_gain, capital_loss, hours_per_week, native_country
    ]], columns=[
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
    ])

    # One-hot encode to match training data
    input_encoded = pd.get_dummies(input_data)

    # Load the training columns and align
    model_columns = joblib.load('model_columns.pkl')
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0]

    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Confidence:** {probability.max():.2%}")

