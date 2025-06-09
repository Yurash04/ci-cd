import os
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    """
    Download model from URL specified in MODEL_URL environment variable.
    """
    try:
        # Create models directory if it doesn't exist
        model_dir = Path('./models/best')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model URL from environment variable
        model_url = os.getenv('MODEL_URL')
        if not model_url:
            raise ValueError("MODEL_URL environment variable is not set")
        
        local_path = model_dir / 'full_model_pipeline.joblib'
        
        logger.info(f"Downloading model from {model_url}")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"Model downloaded successfully to {local_path}")
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    download_model() 