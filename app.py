import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import pandas as pd

# Load trained model and scaler
try:
    model = tf.keras.models.load_model('lstm_stock_model.h5')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"❌ Failed to load model or scaler: {e}")
    st.stop()

# Constants
TIME_STEP = 60

# App UI
st.title("📈 Stock Price Prediction with LSTM")
st.write("Enter the last 60 days of stock closing prices (comma-separated) to predict the next day's closing price.")

# User input
user_input = st.text_area("Enter historical closing prices (comma-separated):")

# On predict button
if st.button("Predict"):
    if not user_input:
        st.warning("⚠️ Please enter historical closing prices.")
    else:
        try:
            # Parse and validate input
            prices = [float(p.strip()) for p in user_input.split(',')]
            if len(prices) != TIME_STEP:
                st.warning(f"⚠️ Exactly {TIME_STEP} values required. You entered {len(prices)}.")
            else:
                # Prepare dummy 5-feature input for scaler
                dummy_data = np.zeros((TIME_STEP, 5))
                dummy_data[:, 0] = prices  # 'Close' assumed as first feature

                # Scale and reshape input
                scaled = scaler.transform(dummy_data)
                input_sequence = scaled[:, 0].reshape(1, TIME_STEP, 1)

                # Predict
                scaled_prediction = model.predict(input_sequence)

                # Prepare dummy data for inverse transformation
                inverse_dummy = np.zeros((1, 5))
                inverse_dummy[:, 0] = scaled_prediction[:, 0]
                predicted_price = scaler.inverse_transform(inverse_dummy)[:, 0]

                # Show result
                st.success(f"💰 Predicted closing price: **${predicted_price[0]:.2f}**")
        except ValueError:
            st.error("❌ Invalid input. Please enter a list of numbers separated by commas.")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
