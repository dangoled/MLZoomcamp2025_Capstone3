"""
Tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from api.main import app
from api.schemas import CarbonFootprintInput, BatchPredictionRequest

class TestAPIEndpoints:
    """Test API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        # Mock the predictor to avoid loading actual models
        with patch('api.main.predictor'):
            from api.main import app
            return TestClient(app)
    
    @patch('api.main.predictor')
    def test_root_endpoint(self, mock_predictor, client):
        """Test root endpoint"""
        mock_predictor.models = {'test': 'model'}
        
        response = client.get("/")
        