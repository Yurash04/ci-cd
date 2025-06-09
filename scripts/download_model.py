import os
import requests
from pathlib import Path
import logging
import re
from urllib.parse import urlparse, parse_qs
import json

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

def download_from_google_drive(file_id, destination):
    """Download a file from Google Drive."""
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning_'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        total_size = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
                    if total_size % (10 * 1024 * 1024) < CHUNK_SIZE:
                        logger.info(f"Downloaded {total_size} bytes")
        
        return total_size

    try:
        # First request to get the confirmation token
        url = f"https://docs.google.com/uc?export=download&id={file_id}"
        logger.info(f"Making initial request to: {url}")
        
        session = requests.Session()
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        # Log response headers for debugging
        logger.info("Initial response headers:")
        for key, value in response.headers.items():
            logger.info(f"  {key}: {value}")
        
        # Check if we need confirmation
        token = get_confirm_token(response)
        if token:
            logger.info("Download requires confirmation, proceeding with token")
            params = {'id': file_id, 'export': 'download', 'confirm': token}
            response = session.get(url, params=params, stream=True)
            response.raise_for_status()
            
            # Log confirmation response headers
            logger.info("Confirmation response headers:")
            for key, value in response.headers.items():
                logger.info(f"  {key}: {value}")
        
        # Save the file
        total_size = save_response_content(response, destination)
        logger.info(f"Download completed, total size: {total_size} bytes")
        return total_size
        
    except Exception as e:
        logger.error(f"Error downloading from Google Drive: {str(e)}")
        raise

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
            total_size = download_from_google_drive(file_id, local_path)
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