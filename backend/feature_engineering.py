import pandas as pd
import numpy as np
from datetime import datetime

def create_features(data):
    """
    Create features from raw car data.
    
    Args:
        data (dict or pd.DataFrame): Dictionary containing car data or DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with created features
    """
    # Convert to DataFrame if input is dict
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()
    
    # Ensure all required columns are present
    required_columns = [
        'year', 'make', 'model', 'trim', 'body', 'transmission',
        'state', 'condition', 'odometer', 'color', 'interior'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing")
    
    # Create features
    features = pd.DataFrame()
    
    # Numerical features
    features['year'] = df['year']
    features['condition'] = df['condition']
    features['odometer'] = df['odometer']
    
    # Create make_model feature
    features['make_model'] = df['make'] + '_' + df['model']
    
    # Categorical features
    categorical_cols = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior']
    for col in categorical_cols:
        features[col] = df[col].astype('category')
    
    return features 