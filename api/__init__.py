"""
Carbon Footprint Prediction API Module
"""

from .main import app
from .schemas import (
    CarbonFootprintInput,
    CarbonFootprintResponse,
    BatchPredictionRequest,
    ModelInfoResponse,
    PredictionExplanation
)
from .predict import CarbonFootprintPredictor

# API Version and Metadata
__version__ = "1.0.0"
__api_title__ = "Carbon Footprint Prediction API"
__api_description__ = "REST API for predicting personal carbon footprint based on daily behavior patterns"
__api_contact__ = {
    "name": "API Support",
    "email": "support@carbonfootprint.ai"
}
__api_license__ = {
    "name": "MIT",
    "url": "https://opensource.org/licenses/MIT"
}

# Export main API components
__all__ = [
    "app",                    # FastAPI application instance
    "CarbonFootprintPredictor",  # Main predictor class
    "CarbonFootprintInput",   # Input validation schema
    "CarbonFootprintResponse", # Response schema
    "BatchPredictionRequest", # Batch prediction schema
    "ModelInfoResponse",      # Model info schema
    "PredictionExplanation"   # Explanation schema
]

# Optional: Add helper functions for common API tasks
def get_api_info():
    """
    Get API information dictionary
    
    Returns:
        Dictionary with API metadata
    """
    return {
        "title": __api_title__,
        "version": __version__,
        "description": __api_description__,
        "contact": __api_contact__,
        "license": __api_license__,
        "endpoints": {
            "root": "/",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict-batch",
            "model_info": "/model-info",
            "explain": "/explain/{prediction_id}",
            "sample_input": "/sample-input"
        }
    }

def create_test_client():
    """
    Create a test client for the API
    
    Returns:
        TestClient instance for testing
    """
    try:
        from fastapi.testclient import TestClient
        return TestClient(app)
    except ImportError:
        raise ImportError("fastapi.testclient is required for testing")

# Initialize package with startup message
print(f"Carbon Footprint Prediction API v{__version__}")
print(f"API available at: http://localhost:8000")
print(f"Documentation: http://localhost:8000/docs")