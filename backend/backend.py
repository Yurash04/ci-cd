import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException
import joblib
import pandas as pd
import numpy as np
import logging
import traceback
import webcolors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from sklearn import set_config
import re
from features import create_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_config(transform_output="pandas")

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

app = FastAPI()

def color_to_rgb(color_name):
    """Convert color name to RGB tuple with error handling"""
    try:
        color_str = str(color_name).lower()
        return webcolors.name_to_rgb(color_str)
    except (ValueError, TypeError):
        return (0, 0, 0)

class FeatureHasherTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=10):
        self.n_features = n_features
        self.hasher = FeatureHasher(n_features=n_features, input_type='string')
        self.col_name = None

    def fit(self, X, y=None):
        self.col_name = X.columns[0]
        return self

    def transform(self, X):
        col_series = X[self.col_name].fillna("")
        hashed = self.hasher.transform([[x] for x in col_series])
        return pd.DataFrame(
            hashed.toarray(),
            columns=[f"{self.col_name}_hash_{i}" for i in range(self.n_features)],
            index=X.index
        )

class MinMaxScalerDF(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.column_mins = {}
        self.column_maxs = {}
        self.column_indices = {}

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            for i, col in enumerate(X.columns):
                if col in self.columns:
                    self.column_indices[col] = i

        for col in self.columns:
            if col in X.columns:
                self.column_mins[col] = X[col].min()
                self.column_maxs[col] = X[col].max()
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for col in self.columns:
                if col in X.columns and col in self.column_mins:
                    col_min = self.column_mins[col]
                    col_max = self.column_maxs[col]
                    if col_max - col_min > 0:
                        X[col] = (X[col] - col_min) / (col_max - col_min)
                    else:
                        X[col] = 0
            return X
        else:
            X = X.copy()
            for col in self.columns:
                if col in self.column_indices:
                    idx = self.column_indices[col]
                    col_min = self.column_mins[col]
                    col_max = self.column_maxs[col]
                    if col_max - col_min > 0:
                        X[:, idx] = (X[:, idx] - col_min) / (col_max - col_min)
                    else:
                        X[:, idx] = 0
            return X

def clean_saledate(date_str):
    """Clean and standardize saledate format"""
    if pd.isna(date_str):
        return None

    match = re.search(r'([A-Za-z]{3} [A-Za-z]{3} \d{2} \d{4} \d{2}:\d{2}:\d{2})', date_str)
    if match:
        return match.group(1)
    return date_str

def preprocess_input(df):
    """Preprocess input data before prediction"""
    required_cols = ['year', 'make', 'model', 'body', 'vin', 'state',
                    'condition', 'odometer', 'color', 'interior',
                    'seller', 'saledate']

    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df['transmission'] = df.get('transmission', 'Unknown')
    df['trim'] = df.get('trim', '')

    df['color'] = df['color'].fillna('Other').replace({'—': 'Other'})
    df['interior'] = df['interior'].fillna('Other').replace({'—': 'Other'})
    df['transmission'] = df['transmission'].fillna('Unknown')

    return df

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/model_info")
async def model_info():
    """Return model metadata including type, best parameters, and metrics"""
    return {
        "model_type": "Random Forest Regressor",
        "best_parameters": {
            "max_depth": None,
            "max_features": "sqrt",
            "n_estimators": 100
        },
        "metrics": {
            "rmse": 7190.26,
            "r2": 0.49
        }
    }

@app.post("/predict")
async def predict(car: dict):
    try:
        raw_df = pd.DataFrame([car])
        processed_df = preprocess_input(raw_df)

        prediction = full_pipeline.predict(processed_df)[0]
        rounded_prediction = round(float(prediction), 2)
        return {"prediction": rounded_prediction}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(file: UploadFile):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(400, detail="Only CSV files are supported")

        try:
            raw_df = pd.read_csv(file.file)
            logger.info(f"Read CSV with {len(raw_df)} rows")
        except Exception as e:
            raise HTTPException(400, detail=f"Invalid CSV: {str(e)}")

        processed_df = preprocess_input(raw_df)

        try:
            predictions = full_pipeline.predict(processed_df)
            rounded_predictions = [round(float(p), 2) for p in predictions]
            return {"predictions": rounded_predictions}
        except Exception as e:
            for i, row in processed_df.iterrows():
                try:
                    test_df = pd.DataFrame([row])
                    full_pipeline.predict(test_df)
                except Exception as row_error:
                    logger.error(f"Error in row {i}: {row_error}\nRow data: {row.to_dict()}")
            raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
