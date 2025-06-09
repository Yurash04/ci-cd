import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time

BACKEND_URL = "http://localhost:8000"

def check_health():
    response = requests.get(f"{BACKEND_URL}/health")
    return response.json()

def get_model_info():
    response = requests.get(f"{BACKEND_URL}/model_info")
    return response.json()

def predict(data):
    response = requests.post(f"{BACKEND_URL}/predict", json=data)
    return response.json()

def predict_batch(file):
    files = {'file': file}
    response = requests.post(f"{BACKEND_URL}/predict_batch", files=files)
    return response.json()

st.title("Car Price Prediction")

with st.expander("System Health", expanded=False):
    if st.button("Check Service Health"):
        with st.spinner("Checking backend health..."):
            try:
                health_status = check_health()
                st.success(f"Backend status: {health_status['status']}")
            except:
                st.error("Backend service unavailable")

with st.expander("Model Information", expanded=False):
    if st.button("Get Model Details"):
        try:
            model_info = get_model_info()
            st.subheader("Model Type")
            st.write(model_info["model_type"])
            
            st.subheader("Best Parameters")
            st.json(model_info["best_parameters"])
            
            st.subheader("Evaluation Metrics")
            st.metric("RMSE", model_info["metrics"]["rmse"])
            st.metric("R² Score", model_info["metrics"]["r2"])
        except:
            st.error("Could not retrieve model information")

with st.expander("Single Car Prediction", expanded=True):
    with st.form("single_prediction_form"):
        st.subheader("Enter Car Details")
        
        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input("Year", min_value=1980, max_value=2025, value=2020)
            make = st.text_input("Make", "Toyota")
            model = st.text_input("Model", "Camry")
            trim = st.text_input("Trim", "LE")
            body = st.selectbox("Body Type", [
                "sedan", "coupe", "van", "convertible", "wagon", 
                "cab", "suv", "hatchback", "minivan", "other"
            ])
            transmission = st.selectbox("Transmission", ["Automatic", "Manual", "CVT", "DCT", "Unknown"])
            
        with col2:
            vin = st.text_input("VIN", "1HGCM82633A123456")
            state = st.text_input("State (Abbreviation)", "CA")
            condition = st.slider("Condition (1-49)", 1, 49, 4)
            odometer = st.number_input("Odometer (miles)", min_value=0, max_value=500000, value=50000)
            color = st.text_input("Exterior Color", "Blue")
            interior = st.text_input("Interior Color", "Black")
            seller = st.text_input("Seller", "Dealer Name")
        
        sale_date = st.date_input("Sale Date", value=datetime.today())
        sale_time = st.time_input("Sale Time", value=datetime.now().time())
        
        submitted = st.form_submit_button("Predict Price")
        
        if submitted:
            if len(state.strip()) != 2:
                st.error("State must be a 2-character abbreviation")
                st.stop()
                
            formatted_date = f"{sale_date.strftime('%a %b %d %Y')} {sale_time.strftime('%H:%M:%S')}"
            
            car_data = {
                "year": year,
                "make": make,
                "model": model,
                "trim": trim,
                "body": body,
                "transmission": transmission,
                "vin": vin,
                "state": state,
                "condition": condition,
                "odometer": odometer,
                "color": color,
                "interior": interior,
                "seller": seller,
                "saledate": formatted_date
            }
            
            try:
                with st.spinner("Predicting price..."):
                    result = predict(car_data)
                    st.success(f"Predicted selling price: ${result['prediction']:,.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

with st.expander("Batch Prediction from CSV", expanded=True):
    st.subheader("Upload CSV File")
    st.info("CSV must contain these columns: year, make, model, trim, body, transmission, vin, state, condition, odometer, color, interior, seller, saledate")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head(3))
            
            if st.button("Predict Prices"):
                with st.spinner("Processing batch prediction..."):
                    uploaded_file.seek(0)
                    
                    try:
                        result = predict_batch(uploaded_file)
                        
                        df['sellingprice'] = result['predictions']
                        
                        st.subheader("Prediction Results")
                        st.dataframe(df.head(3))
                        
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name='car_price_predictions.csv',
                            mime='text/csv'
                        )
                    except Exception as e:
                        st.error(f"Batch prediction failed: {str(e)}")
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")