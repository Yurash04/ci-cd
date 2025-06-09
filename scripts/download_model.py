import os
import requests
from pathlib import Path
import logging
import re
from urllib.parse import urlparse, parse_qs
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_direct_download_url(yandex_url):
    """
    Convert Yandex.Disk public URL to direct download URL
    """
    try:
        # Extract file ID from Yandex.Disk URL
        if 'disk.yandex.ru' in yandex_url:
            # For public share links
            response = requests.get(yandex_url)
            response.raise_for_status()
            
            # Try to find the download URL in the page
            # First, try to find the data-react-props attribute
            react_props_match = re.search(r'data-react-props="([^"]+)"', response.text)
            if react_props_match:
                try:
                    # Decode HTML entities and parse JSON
                    import html
                    props_json = html.unescape(react_props_match.group(1))
                    props = json.loads(props_json)
                    
                    # Try to find download URL in different possible locations
                    if 'downloadUrl' in props:
                        return props['downloadUrl']
                    elif 'publicUrl' in props:
                        return props['publicUrl']
                    elif 'resource' in props and 'downloadUrl' in props['resource']:
                        return props['resource']['downloadUrl']
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON from data-react-props")
            
            # If we couldn't find the URL in data-react-props, try other methods
            # Look for the download button URL
            download_url_match = re.search(r'href="(https://[^"]+download[^"]+)"', response.text)
            if download_url_match:
                return download_url_match.group(1)
            
            # Look for any URL containing 'download'
            download_url_match = re.search(r'https://[^"\']+download[^"\']+', response.text)
            if download_url_match:
                return download_url_match.group(0)
            
            # If we still haven't found the URL, try to get it from the page source
            logger.info("Trying to find download URL in page source...")
            logger.debug(f"Page content: {response.text[:1000]}...")  # Log first 1000 chars for debugging
            
            raise ValueError("Could not find download URL in Yandex.Disk page")
        else:
            # If it's already a direct download URL
            return yandex_url
            
    except Exception as e:
        logger.error(f"Error getting direct download URL: {str(e)}")
        raise

def download_model():
    """
    Download model from URL specified in MODEL_URL environment variable.
    Supports both direct download URLs and Yandex.Disk public URLs.
    """
    try:
        # Create models directory if it doesn't exist
        model_dir = Path('./models/best')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model URL from environment variable
        model_url = os.getenv('MODEL_URL')
        if not model_url:
            raise ValueError("MODEL_URL environment variable is not set")
        
        logger.info(f"Original URL: {model_url}")
        
        # Get direct download URL if it's a Yandex.Disk link
        download_url = get_direct_download_url(model_url)
        logger.info(f"Using download URL: {download_url}")
        
        local_path = model_dir / 'full_model_pipeline.joblib'
        
        logger.info(f"Downloading model from {download_url}")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Get file size from headers
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Expected file size: {total_size} bytes")
        
        with open(local_path, 'wb') as f:
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Log progress every 10MB
                    if downloaded_size % (10 * 1024 * 1024) < 8192:
                        logger.info(f"Downloaded {downloaded_size}/{total_size} bytes")
        
        # Verify file size
        actual_size = local_path.stat().st_size
        if actual_size != total_size:
            raise ValueError(f"Downloaded file size {actual_size} does not match expected size {total_size}")
                
        logger.info(f"Model downloaded successfully to {local_path}")
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    download_model() 