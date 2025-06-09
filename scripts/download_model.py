import os
import boto3
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    """
    Download model from S3 bucket.
    Requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.
    """
    try:
        # Create models directory if it doesn't exist
        model_dir = Path('./models/best')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client
        s3 = boto3.client('s3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # Download model file
        bucket_name = os.getenv('AWS_BUCKET_NAME', 'your-bucket-name')
        model_key = 'models/full_model_pipeline.joblib'
        local_path = model_dir / 'full_model_pipeline.joblib'
        
        logger.info(f"Downloading model from s3://{bucket_name}/{model_key}")
        s3.download_file(bucket_name, model_key, str(local_path))
        logger.info(f"Model downloaded successfully to {local_path}")
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    download_model() 