import tensorflow as tf
import streamlit as st
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

# Time step used during model training
time_step = 60

# Streamlit App Title and Instructions
st.title("📈 Stock Price Prediction with LSTM")
st.write("Enter the last 60 days of historical stock closing prices (comma-separated) to predict the next day's closing price.")

# Input area for user to paste historical prices
historical_prices_input = st.text_area("Enter historical closing prices (comma-separated):")

# Predict button logic
if st.button("Predict"):
    if historical_prices_input:
        try:
            # Convert input string to float list
            historical_prices = np.array([float(price.strip()) for price in historical_prices_input.split(',')])

            if len(historical_prices) != time_step:
                st.warning(f"⚠️ Please enter exactly {time_step} historical closing prices.")
            else:
                # Create dummy array for 5 features; only 'Close' will be filled
                dummy_data = np.zeros((time_step, 5))
                dummy_data[:, 0] = historical_prices  # Assuming 'Close' was the first feature

                # Scale input
                scaled_input = scaler.transform(dummy_data)

                # Reshape for LSTM: (samples, timesteps, features)
                scaled_input = scaled_input[:, 0].reshape(1, time_step, 1)

                # Make prediction
                scaled_prediction = model.predict(scaled_input)

                # Create dummy for inverse scaling
                dummy_prediction_data = np.zeros((scaled_prediction.shape[0], 5))
                dummy_prediction_data[:, 0] = scaled_prediction[:, 0]

                # Inverse transform to get actual predicted price
                predicted_price = scaler.inverse_transform(dummy_prediction_data)[:, 0]

                # Display prediction
                st.success(f"💹 Predicted next day's closing price: **${predicted_price[0]:.2f}**")

        except ValueError:
            st.error("❌ Invalid input. Please enter a comma-separated list of numbers.")
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")
    else:
        st.warning("⚠️ Please enter historical closing prices.")

# Instructions for running the app
st.markdown("""
---

