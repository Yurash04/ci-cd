from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher

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