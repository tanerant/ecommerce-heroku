# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 00:14:52 2021

@author: TNR
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


st.write("""
         
# Shipment Delivered on time or not?

This application makes a simple estimate of whether the order is delivered on time or not.

Data source : https://www.kaggle.com/prachi13/customer-analytics 
""") 

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/tanerant/ecommerce-heroku/main/ecommerce_example.csv)
""")


uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Warehouse_block = st.sidebar.selectbox('Warehouse_block',('A','B','C','D','E','F'))
        Gender = st.sidebar.selectbox('Gender',('M','F'))
        Mode_of_Shipment = st.sidebar.selectbox('Mode_of_Shipment',('Ship','Flight','Road'))
        Product_importance = st.sidebar.selectbox('Product_importance',('low','medium','high'))
        Customer_care_calls = st.sidebar.slider('Customer_care_calls', 2,7,5)
        Customer_rating = st.sidebar.slider('Customer_rating', 1,5,3)
        Cost_of_the_Product = st.sidebar.slider('Cost_of_the_Product', 96,310,200)
        Prior_purchases = st.sidebar.slider('Prior_purchases', 2,10,5)
        Discount_offered = st.sidebar.slider('Discount_offered', 1,65,10)
        Weight_in_gms = st.sidebar.slider('Weight_in_gms', 1001,7846,3000)
        
        
        data = {'Warehouse_block': Warehouse_block,
                'Mode_of_Shipment': Mode_of_Shipment,
                'Product_importance': Product_importance,
                'Customer_care_calls': Customer_care_calls,
                'Customer_rating': Customer_rating,
                'Cost_of_the_Product': Cost_of_the_Product,
                'Prior_purchases': Prior_purchases,
                'Discount_offered': Discount_offered,
                'Weight_in_gms': Weight_in_gms,
                'Gender': Gender}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
    
    
    ecommerce_raw = pd.read_csv('Train.csv')
ecommerce = ecommerce_raw.drop(columns=['Reached.on.Time_Y.N','ID'])

df = pd.concat([input_df,ecommerce],axis=0)

encode = ['Gender','Product_importance','Mode_of_Shipment','Warehouse_block']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1]

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)


# Reads in saved classification model
load_clf = pickle.load(open('ecommerce_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
reached_on_time = np.array(['Reached','Not_Reached'])
st.write(reached_on_time[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
