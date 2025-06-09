import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_model_report():
    try:
        # Load the model pipeline
        model_path = Path('./models/best/full_model_pipeline.joblib')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        pipeline = joblib.load(model_path)
        
        # Load sample data
        sample_data = pd.read_csv('sample_2.csv')
        
        # Make predictions
        predictions = pipeline.predict(sample_data)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(sample_data['sellingprice'], predictions))
        r2 = r2_score(sample_data['sellingprice'], predictions)
        
        # Generate report
        report = f"""# Model Performance Report

## Model Information
- Model Type: Random Forest Regressor
- Training Data Size: {len(sample_data)} samples

## Performance Metrics
- RMSE: {rmse:.2f}
- R² Score: {r2:.2f}

## Feature Importance
{get_feature_importance(pipeline)}

## Sample Predictions
{pd.DataFrame({
    'Actual': sample_data['sellingprice'].head(5),
    'Predicted': predictions[:5]
}).to_markdown()}
"""
        
        # Save report
        with open('model_report.md', 'w') as f:
            f.write(report)
            
        logger.info("Model report generated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error generating model report: {str(e)}")
        return False

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
