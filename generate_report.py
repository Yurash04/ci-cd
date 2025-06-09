import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
from pathlib import Path
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.feature_engineering import create_features
from backend.transformers import FeatureHasherTransformer, MinMaxScalerDF

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_model_report():
    """Generate a report about the model's performance."""
    try:
        # Load the model pipeline
        model_path = Path('./models/best/full_model_pipeline.joblib')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        pipeline = joblib.load(model_path)
        logger.info("Model pipeline loaded successfully")
        
        # Generate sample data for testing
        sample_data = {
            'year': 2015,
            'make': 'Toyota',
            'model': 'Camry',
            'trim': 'LE',
            'body': 'sedan',
            'transmission': 'automatic',
            'vin': '4T1BF1FKXEU123456',
            'state': 'CA',
            'condition': 4,
            'odometer': 50000,
            'color': 'blue',
            'interior': 'black',
            'seller': 'test',
            'saledate': 'Mon Mar 15 2021 12:00:00'
        }
        
        # Create features
        features = create_features(sample_data)
        logger.info("Features created successfully")
        
        # Make prediction
        prediction = pipeline.predict(features)
        logger.info(f"Sample prediction: ${prediction[0]:,.2f}")
        
        # Generate report
        report = f"""# Model Performance Report

## Model Information
- Model Type: {type(pipeline).__name__}
- Features Used: {', '.join(features.columns)}
- Sample Prediction: ${prediction[0]:,.2f}

## Model Pipeline Steps
{pipeline.named_steps}

## Sample Input
```json
{sample_data}
```

## Notes
- This is a sample report generated for demonstration purposes
- The model is trained to predict car prices based on various features
- The sample prediction shows how the model would price a 2015 Toyota Camry LE
"""
        
        # Save report
        report_path = Path('model_report.md')
        report_path.write_text(report)
        logger.info(f"Report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Error generating model report: {str(e)}")
        raise

def get_feature_importance(pipeline):
    try:
        # Get feature names and importance from the pipeline
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        importance = pipeline.named_steps['regressor'].feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df.head(10).to_markdown()
    except:
        return "Feature importance not available"

if __name__ == "__main__":
    generate_model_report()
