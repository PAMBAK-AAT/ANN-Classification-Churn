


import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder



## This block is used to make predictions using our trained model on new data.
model=load_model("model.h5") # Loads your trained neural network from disk

## During training we have map Male->0 and Female->1, so in order to maintain same mapping we have import the encoder
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender=pickle.load(file)


with open('onehot_encoded_geo.pkl', 'rb') as file:
    onehot_encoded_geo=pickle.load(file)

# During training our model learns on the scaled value not raw ones, so we have to use the same scaler
with open('scaler.pkl', 'rb') as file:
    scaler=pickle.load(file)


## streamlit app
st.title('Customer Churn Prediction')

## User input
# onehot_encoded_geo.categories_ : [array(['France', 'Germany', 'Spain'], dtype=object)]
# categories_[0] : ['France', 'Germany', 'Spain']

geography =st.selectbox('Geography', onehot_encoded_geo.categories_[0]) # categories_ stores all unique categories learned during training
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
estimated_salary = st.number_input('Estimated Salary')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit card', [0, 1])
is_active_member = st.selectbox('Is Active member', [0, 1])


## Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


geo_encoded = onehot_encoded_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoded_geo.get_feature_names_out(['Geography']))

# Combine one hot encoded columns with input_data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

## Prediction Churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0] # For neural networks / Keras / TensorFlow models, predict() always returns a 2D array, even for a single input.

st.write(f"Churn probability: {prediction_prob:.2f}.")

if prediction_prob > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')






