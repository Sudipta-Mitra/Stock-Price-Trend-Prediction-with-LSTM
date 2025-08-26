import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import pandas as pd

# Load the trained model and scaler
try:
    model = tf.keras.models.load_model('lstm_stock_model.h5')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Define the time step used during training
time_step = 60 # Make sure this matches the time_step used in your training

# Streamlit App Title and Description
st.title("Stock Price Prediction with LSTM")
st.write("Enter the last 60 days of historical stock prices (separated by commas) to predict the next day's closing price.")

# Input text area for historical prices
historical_prices_input = st.text_area("Enter historical closing prices (comma-separated):")

# Predict button
if st.button("Predict"):
    if historical_prices_input:
        try:
            # Preprocess user input
            historical_prices = np.array([float(price.strip()) for price in historical_prices_input.split(',')])

            if len(historical_prices) != time_step:
                st.warning(f"Please enter exactly {time_step} historical closing prices.")
            else:
                # The scaler was fitted on 5 features (Close, High, Low, Open, Volume).
                # We need to create a dummy array with 5 features for scaling, even though we only have Close prices.
                # Assuming 'Close' was the first column during scaling.
                dummy_data = np.zeros((time_step, 5))
                dummy_data[:, 0] = historical_prices
                scaled_input = scaler.transform(dummy_data)

                # Reshape for LSTM (samples, timesteps, features)
                scaled_input = scaled_input[:, 0].reshape(1, time_step, 1)

                # Make prediction
                scaled_prediction = model.predict(scaled_input)

                # Inverse transform prediction
                # Create a dummy array with the same number of features as the original data for inverse transformation
                dummy_prediction_data = np.zeros((scaled_prediction.shape[0], 5))
                dummy_prediction_data[:, 0] = scaled_prediction[:, 0]
                predicted_price = scaler.inverse_transform(dummy_prediction_data)[:, 0]


                st.success(f"Predicted next day's closing price: {predicted_price[0]:.2f}")

        except ValueError:
            st.error("Invalid input. Please enter a comma-separated list of numbers.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter historical closing prices.")

st.markdown("""
### How to Run the Streamlit App:

1. Save the trained model (`lstm_stock_model.h5`) and the scaler (`scaler.pkl`) in the same directory as this `app.py` file.
2. Open a terminal or command prompt.
3. Navigate to the directory where you saved the files.
4. Run the command: `streamlit run app.py`
""")
