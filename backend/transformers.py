from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class FeatureHasherTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that applies feature hashing to categorical features.
    """
    def __init__(self, n_features=1000):
        self.n_features = n_features
        self.hasher = FeatureHasher(n_features=n_features, input_type='string')
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Convert categorical columns to string type
        X_str = X.astype(str)
        
        # Convert DataFrame to list of dicts for feature hashing
        X_dict = X_str.to_dict('records')
        
        # Apply feature hashing
        X_hashed = self.hasher.transform(X_dict)
        
        return X_hashed

class MinMaxScalerDF(BaseEstimator, TransformerMixin):
    """
    Custom transformer that applies MinMaxScaler while preserving DataFrame structure.
    """
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_names = None
        self.is_dataframe = False
        
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.is_dataframe = True
            self.feature_names = X.columns
            self.scaler.fit(X)
        else:
            self.is_dataframe = False
            self.scaler.fit(X)
        return self
        
    def transform(self, X):
        if not hasattr(self, 'scaler'):
            self.scaler = MinMaxScaler()
            self.fit(X)
            
        if self.is_dataframe:
            X_scaled = self.scaler.transform(X)
            return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        return self.scaler.transform(X)
        
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X) 