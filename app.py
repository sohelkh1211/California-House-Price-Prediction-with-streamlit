import streamlit as st
import numpy as np
from predict_cost.py import predict
from house.py import R2

st.title('House price prediction')
 
st.write('---')

longitute = st.number_input('Enter Longitute of house')
latitude = st.number_input('Enter Latitude of house')

age = st.number_input('How old is the house (in years)?', min_value=0, step=1)

rooms = st.number_input('Total number of rooms')
 
median_income = st.number_input('Median income of house')

ocean_proximity = st.selectbox(
    'Specify Ocean proximity ?',
    ('Choose your Option', '<1H OCEAN', 'INLAND','ISLAND','NEAR BAY','NEAR OCEAN'))

if ocean_proximity == '<1H OCEAN':
  OCEAN = 1.0
  INLAND = 0.0
  ISLAND = 0.0
  NEAR_BAY = 0.0
  NEAR_OCEAN = 0.0
elif ocean_proximity == 'INLAND':
  OCEAN = 0.0
  INLAND = 1.0
  ISLAND = 0.0
  NEAR_BAY = 0.0
  NEAR_OCEAN = 0.0
elif ocean_proximity == 'ISLAND':
  OCEAN = 0.0
  INLAND = 0.0
  ISLAND = 1.0
  NEAR_BAY = 0.0
  NEAR_OCEAN = 0.0
elif ocean_proximity == 'NEAR BAY':
  OCEAN = 0.0
  INLAND = 0.0
  ISLAND = 0.0
  NEAR_BAY = 1.0
  NEAR_OCEAN = 0.0
elif ocean_proximity == 'NEAR OCEAN':
  OCEAN = 0.0
  INLAND = 0.0
  ISLAND = 0.0
  NEAR_BAY = 0.0
  NEAR_OCEAN = 1.0

bedroom_ratio = st.number_input('Total number of bedrooms per room')

household_rooms = st.number_input('Total number of rooms per household')
 
if st.button('Predict House Price'):
    cost = predict(np.array([[longitute, latitude, age, rooms, median_income, OCEAN, INLAND, ISLAND, NEAR_BAY, NEAR_OCEAN, bedroom_ratio, household_rooms]]))
    st.metric(label = "Predicted house price in $ ", value = cost[0])
if st.button('Check Accuracy'):
    accuracy = str(R2*100) + "%"
    st.metric(label="Accuracy", value=accuracy)
