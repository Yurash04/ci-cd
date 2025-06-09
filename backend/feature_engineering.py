import pandas as pd
import numpy as np
from datetime import datetime

def create_features(data):
    """
    Create features from raw car data.
    
    Args:
        data (dict): Dictionary containing car data
        
    Returns:
        pd.DataFrame: DataFrame with created features
    """
    # Convert single sample to DataFrame
    df = pd.DataFrame([data])
    
    # Convert saledate to datetime
    df['saledate'] = pd.to_datetime(df['saledate'])
    
    # Extract date features
    df['sale_year'] = df['saledate'].dt.year
    df['sale_month'] = df['saledate'].dt.month
    df['sale_day'] = df['saledate'].dt.day
    df['sale_dayofweek'] = df['saledate'].dt.dayofweek
    
    # Calculate car age
    df['car_age'] = df['sale_year'] - df['year']
    
    # Create categorical features
    categorical_cols = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    # Drop original date column and other non-feature columns
    df = df.drop(['saledate', 'vin', 'seller'], axis=1)
    
    return df 