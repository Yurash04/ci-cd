import os
import logging
from pathlib import Path
import requests
import re
import gdown

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_google_drive_file_id(url):
    """Extract file ID from Google Drive URL."""
    logger.info(f"Checking if URL is Google Drive format: {url[:50]}...")
    
    # Extract ID from masked URL
    # Format: ***id=10d... -> 10d...
    if 'id=' in url:
        file_id = url.split('id=')[1]
        logger.info(f"Found Google Drive file ID from masked URL: {file_id}")
        return file_id
    
    # Check full URL patterns
    patterns = [
        r'https://drive\.google\.com/file/d/([^/]+)',
        r'https://drive\.google\.com/open\?id=([^&]+)',
        r'https://docs\.google\.com/uc\?id=([^&]+)',
        r'https://drive\.google\.com/uc\?id=([^&]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            file_id = match.group(1)
            logger.info(f"Found Google Drive file ID from full URL: {file_id}")
            return file_id
    
    logger.info("URL is not in Google Drive format")
    return None

def download_model():
    """
    Download model from URL specified in MODEL_URL environment variable.
    Supports both direct download URLs and Google Drive URLs.
    """
    try:
        # Create models directory if it doesn't exist
        model_dir = Path('./models/best')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model URL from environment variable
        model_url = os.getenv('MODEL_URL')
        if not model_url:
            raise ValueError("MODEL_URL environment variable is not set")
        
        logger.info(f"Original URL: {model_url[:50]}...")
        local_path = model_dir / 'full_model_pipeline.joblib'
        
        # Check if it's a Google Drive URL
        file_id = get_google_drive_file_id(model_url)
        if file_id:
            logger.info(f"Detected Google Drive URL, file ID: {file_id}")
            # Use gdown to download the file
            url = f"https://drive.google.com/uc?id={file_id}"
            logger.info(f"Downloading from: {url}")
            gdown.download(url, str(local_path), quiet=False)
            
            # Verify file size
            actual_size = local_path.stat().st_size
            expected_size = 147_990_000  # примерный размер в байтах
            if actual_size < expected_size:
                raise ValueError(f"Downloaded file size {actual_size} is less than expected {expected_size}")
            
            logger.info(f"Model downloaded successfully to {local_path}")
        else:
            # Direct download
            logger.info(f"Using direct download URL: {model_url[:50]}...")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            # Log response headers for debugging
            logger.info("Response headers:")
            for key, value in response.headers.items():
                logger.info(f"  {key}: {value}")
            
            total_size = int(response.headers.get('content-length', 0))
            logger.info(f"Expected file size: {total_size} bytes")
            
            with open(local_path, 'wb') as f:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if downloaded_size % (10 * 1024 * 1024) < 8192:
                            logger.info(f"Downloaded {downloaded_size}/{total_size} bytes")
                total_size = downloaded_size
            
            # Verify file size
            actual_size = local_path.stat().st_size
            expected_size = 147_990_000  # примерный размер в байтах
            if actual_size < expected_size:
                raise ValueError(f"Downloaded file size {actual_size} is less than expected {expected_size}")
            
            logger.info(f"Model downloaded successfully to {local_path}")
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    download_model() 