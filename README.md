# Car Price Prediction Service

This project provides a machine learning service for predicting car prices based on various features. It consists of a FastAPI backend service and a Streamlit frontend application.

## Project Structure

```
.
├── backend/             # FastAPI backend service
├── frontend/           # Streamlit frontend application
├── models/             # Trained machine learning models
├── docker-compose.yml  # Docker compose configuration
└── requirements.txt    # Python dependencies
```

## Model Report

The latest model performance report can be found [here](model_report.md).

## Setup

### 1. Download the Model

The model file is stored in AWS S3. To download it:

1. Set up AWS credentials:
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_BUCKET_NAME=your_bucket_name
   ```

2. Run the download script:
   ```bash
   python scripts/download_model.py
   ```

### 2. Running the Application

#### Using Docker Compose

1. Make sure you have Docker and Docker Compose installed
2. Run the following command in the project root:
   ```bash
   docker-compose up --build
   ```
3. Access the frontend at http://localhost:8501
4. The backend API will be available at http://localhost:8000

#### Manual Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the backend service:
   ```bash
   python backend/backend.py
   ```

3. Start the frontend application:
   ```bash
   streamlit run frontend/frontend.py
   ```

## API Endpoints

- `GET /health` - Health check endpoint
- `GET /model_info` - Get model metadata and performance metrics
- `POST /predict` - Predict price for a single car
- `POST /predict_batch` - Predict prices for multiple cars from a CSV file

## Development

### Running Tests

The CI pipeline automatically tests the backend service by:
1. Starting the backend server
2. Making a test prediction
3. Verifying the response

### Building Docker Images

Docker images are automatically built and pushed to DockerHub when changes are pushed to the main branch. The images are:
- `car-price-backend:latest`
- `car-price-frontend:latest`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.
