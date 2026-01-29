"""
Pytest configuration and fixtures
"""

import pytest
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import CarbonFootprintPreprocessor
from src.predict import ModelPredictor
from api.schemas import CarbonFootprintInput

@pytest.fixture
def sample_input_data():
    """Sample input data for testing"""
    return {
        'day_type': 'Weekday',
        'transport_mode': 'Car',
        'distance_km': 15.5,
        'electricity_kwh': 8.2,
        'renewable_usage_pct': 25.0,
        'food_type': 'Mixed',
        'screen_time_hours': 6.5,
        'waste_generated_kg': 0.75,
        'eco_actions': 2
    }

@pytest.fixture
def sample_dataframe():
    """Sample data as DataFrame"""
    data = {
        'day_type': ['Weekday', 'Weekend', 'Weekday'],
        'transport_mode': ['Car', 'Bike', 'Bus'],
        'distance_km': [15.5, 5.0, 12.0],
        'electricity_kwh': [8.2, 3.5, 7.0],
        'renewable_usage_pct': [25.0, 80.0, 40.0],
        'food_type': ['Mixed', 'Veg', 'Non-Veg'],
        'screen_time_hours': [6.5, 3.0, 5.0],
        'waste_generated_kg': [0.75, 0.3, 0.6],
        'eco_actions': [2, 5, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_predictions():
    """Sample prediction results"""
    return [
        {
            'carbon_footprint_kg': 10.25,
            'carbon_impact_level': 'High',
            'confidence': 0.85,
            'suggestions': [
                {'category': 'Transport', 'action': 'Switch to public transport'}
            ]
        },
        {
            'carbon_footprint_kg': 4.75,
            'carbon_impact_level': 'Low',
            'confidence': 0.92,
            'suggestions': [
                {'category': 'Maintenance', 'action': 'Continue good practices'}
            ]
        }
    ]

@pytest.fixture
def mock_models():
    """Mock models for testing"""
    models_dir = Path(__file__).parent.parent / 'models'
    
    # Create mock preprocessor
    preprocessor = Mock()
    preprocessor.transform.return_value = np.array([[1, 2, 3, 4, 5]])
    
    # Create mock regression model
    regression_model = Mock()
    regression_model.predict.return_value = np.array([8.25])
    
    # Create mock classification model
    classification_model = Mock()
    classification_model.predict.return_value = np.array([1])  # Medium
    classification_model.predict_proba.return_value = np.array([[0.2, 0.7, 0.1]])
    
    return {
        'preprocessor': preprocessor,
        'regression': regression_model,
        'classification': classification_model
    }

@pytest.fixture
def test_predictor(tmp_path):
    """Create a test predictor with mocked models"""
    # Create temporary models directory
    models_dir = tmp_path / 'models'
    models_dir.mkdir()
    
    # Create mock model files
    with open(models_dir / 'preprocessor.pkl', 'wb') as f:
        pickle.dump(Mock(), f)
    
    with open(models_dir / 'best_regression_model.pkl', 'wb') as f:
        pickle.dump(Mock(), f)
    
    with open(models_dir / 'best_classification_model.pkl', 'wb') as f:
        pickle.dump(Mock(), f)
    
    with open(models_dir / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(Mock(), f)
    
    # Create test metrics
    metrics = {
        'best_models': {'regression': 'random_forest', 'classification': 'random_forest'},
        'regression': {'random_forest': {'test_r2': 0.72}},
        'classification': {'random_forest': {'test_accuracy': 0.81}}
    }
    
    with open(models_dir / 'training_results.json', 'w') as f:
        json.dump(metrics, f)
    
    # Create predictor
    with patch('src.predict.ModelPredictor._load_models'):
        predictor = ModelPredictor(str(models_dir))
        
        # Mock the actual models
        predictor.preprocessor = Mock()
        predictor.preprocessor.transform.return_value = np.array([[1, 2, 3, 4, 5]])
        
        predictor.models = {
            'best_regression': Mock(),
            'best_classification': Mock()
        }
        
        predictor.models['best_regression'].predict.return_value = np.array([8.25])
        predictor.models['best_classification'].predict.return_value = np.array([1])
        predictor.models['best_classification'].predict_proba.return_value = np.array([[0.2, 0.7, 0.1]])
        
        predictor.label_encoder = Mock()
        predictor.label_encoder.inverse_transform.return_value = np.array(['Medium'])
        predictor.label_encoder.classes_ = np.array(['Low', 'Medium', 'High'])
        
        predictor.metrics = metrics
    
    return predictor

@pytest.fixture
def test_client():
    """Create a test client for API testing"""
    # Mock the predictor to avoid loading actual models
    with patch('api.main.predictor'):
        from api.main import app
        from fastapi.testclient import TestClient
        return TestClient(app)

@pytest.fixture
def invalid_inputs():
    """Invalid input data for testing validation"""
    return [
        # Missing required field
        {
            'transport_mode': 'Car',
            'distance_km': 15.5,
            # Missing day_type
        },
        # Invalid categorical value
        {
            'day_type': 'InvalidDay',
            'transport_mode': 'Car',
            'distance_km': 15.5,
            'electricity_kwh': 8.2,
            'renewable_usage_pct': 25.0,
            'food_type': 'Mixed',
            'screen_time_hours': 6.5,
            'waste_generated_kg': 0.75,
            'eco_actions': 2
        },
        # Out of range numeric value
        {
            'day_type': 'Weekday',
            'transport_mode': 'Car',
            'distance_km': 150.0,  # Too high
            'electricity_kwh': 8.2,
            'renewable_usage_pct': 25.0,
            'food_type': 'Mixed',
            'screen_time_hours': 6.5,
            'waste_generated_kg': 0.75,
            'eco_actions': 2
        },
        # Negative value
        {
            'day_type': 'Weekday',
            'transport_mode': 'Car',
            'distance_km': 15.5,
            'electricity_kwh': -5.0,  # Negative
            'renewable_usage_pct': 25.0,
            'food_type': 'Mixed',
            'screen_time_hours': 6.5,
            'waste_generated_kg': 0.75,
            'eco_actions': 2
        }
    ]

@pytest.fixture
def edge_case_inputs():
    """Edge case inputs for testing"""
    return [
        # Minimum values
        {
            'day_type': 'Weekday',
            'transport_mode': 'Walk',
            'distance_km': 0.0,
            'electricity_kwh': 0.0,
            'renewable_usage_pct': 0.0,
            'food_type': 'Veg',
            'screen_time_hours': 0.0,
            'waste_generated_kg': 0.0,
            'eco_actions': 0
        },
        # Maximum values
        {
            'day_type': 'Weekend',
            'transport_mode': 'Car',
            'distance_km': 100.0,
            'electricity_kwh': 50.0,
            'renewable_usage_pct': 100.0,
            'food_type': 'Non-Veg',
            'screen_time_hours': 24.0,
            'waste_generated_kg': 10.0,
            'eco_actions': 10
        },
        # Mixed extreme values
        {
            'day_type': 'Weekday',
            'transport_mode': 'EV',
            'distance_km': 0.5,
            'electricity_kwh': 20.0,
            'renewable_usage_pct': 100.0,
            'food_type': 'Mixed',
            'screen_time_hours': 12.0,
            'waste_generated_kg': 0.1,
            'eco_actions': 10
        }
    ]

# Pytest configuration
def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )

def pytest_collection_modifyitems(config, items):
    """Skip tests based on command line options"""
    skip_integration = pytest.mark.skip(reason="Skipping integration test")
    skip_slow = pytest.mark.skip(reason="Skipping slow test")
    
    for item in items:
        if "integration" in item.keywords and not config.getoption("--integration"):
            item.add_marker(skip_integration)
        if "slow" in item.keywords and not config.getoption("--slow"):
            item.add_marker(skip_slow)