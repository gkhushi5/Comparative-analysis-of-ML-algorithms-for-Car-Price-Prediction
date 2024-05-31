import numpy as np
import streamlit as st
import joblib

# loading the model and scaler
model = joblib.load('C:/Users/dell/Projects/Minor Project/xgb_model.pkl')
scaler = joblib.load('C:/Users/dell/Projects/Minor Project/scaler.pkl')

def car_price_prediction(input_data):
    #Changing the input into numpy array and reshaping
    input_changed = np.array(input_data).reshape(1,-1)

    #Standardize the input
    std_input = scaler.transform(input_changed)

    prediction = model.predict(std_input)

    return "Estimated Car Price - " + str(prediction[0])

def main():
    #creating the title
    st.title('Car Price Predictions')

    #Getting the input from user 
    year = st.number_input('Year',step=1)
    transmission = st.number_input('Transmission : Automatic:0, Manual:1, Semi-Auto:2',step=1)
    mileage = st.text_input('Mileage')
    fuel_type = st.number_input('Fuel Type: 0-Petrol, 1-Diesel,2- Hybrid, 3-Electric, 4-Other',step=1)
    tax = st.number_input('Tax',step=1)
    mpg = st.text_input('MPG')
    enginesize = st.text_input('Engine Size')

    pred_price = ''

    #Create a button 
    if st.button('Check Estimated Price'):
   
        pred_price = car_price_prediction([year, transmission, mileage, fuel_type, tax, mpg, enginesize])
  
    st.success(pred_price)


if __name__ == '__main__':
    main()