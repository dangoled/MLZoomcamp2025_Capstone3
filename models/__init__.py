"""
Models module for Carbon Footprint Prediction
"""

import os
from pathlib import Path
import json
import pickle
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

MODELS_DIR = Path(__file__).parent

def list_available_models() -> List[str]:
    """
    List all available models in the models directory
    
    Returns:
        List of available model files
    """
    model_files = []
    for file in MODELS_DIR.glob("*.pkl"):
        model_files.append(file.name)
    return sorted(model_files)

def load_model(model_name: str):
    """
    Load a model from the models directory
    
    Args:
        model_name: Name of the model file (e.g., 'best_regression_model.pkl')
    
    Returns:
        Loaded model object
    """
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_preprocessor():
    """Load the preprocessing pipeline"""
    return load_model('preprocessor.pkl')

def load_label_encoder():
    """Load the label encoder"""
    return load_model('label_encoder.pkl')

def load_best_regression_model():
    """Load the best regression model"""
    return load_model('best_regression_model.pkl')

def load_best_classification_model():
    """Load the best classification model"""
    return load_model('best_classification_model.pkl')

def get_training_results() -> Dict[str, Any]:
    """
    Load training results and metrics
    
    Returns:
        Dictionary with training results
    """
    results_path = MODELS_DIR / 'training_results.json'
    if not results_path.exists():
        return {}
    
    with open(results_path, 'r') as f:
        return json.load(f)

def get_model_info() -> Dict[str, Any]:
    """
    Get information about all available models
    
    Returns:
        Dictionary with model information
    """
    info = {
        'available_models': list_available_models(),
        'training_results': get_training_results(),
        'models_directory': str(MODELS_DIR.absolute())
    }
    
    # Add model statistics if available
    results = get_training_results()
    if 'best_models' in results:
        info['best_models'] = results['best_models']
    
    if 'dataset_info' in results:
        info['dataset_info'] = results['dataset_info']
    
    return info

def validate_models() -> Dict[str, bool]:
    """
    Validate that all required models exist and are loadable
    
    Returns:
        Dictionary with validation results for each model
    """
    validation_results = {}
    required_models = [
        'preprocessor.pkl',
        'label_encoder.pkl',
        'best_regression_model.pkl',
        'best_classification_model.pkl'
    ]
    
    for model_file in required_models:
        model_path = MODELS_DIR / model_file
        exists = model_path.exists()
        
        if exists:
            # Try to load the model to ensure it's valid
            try:
                load_model(model_file)
                loadable = True
                error = None
            except Exception as e:
                loadable = False
                error = str(e)
        else:
            loadable = False
            error = "File not found"
        
        validation_results[model_file] = {
            'exists': exists,
            'loadable': loadable,
            'error': error if not loadable else None
        }
    
    return validation_results

def get_sample_input() -> Dict[str, Any]:
    """
    Get a sample input for testing predictions
    
    Returns:
        Sample input dictionary
    """
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

def get_sample_predictions() -> List[Dict[str, Any]]:
    """
    Get sample predictions for testing
    
    Returns:
        List of sample predictions
    """
    samples_path = MODELS_DIR / 'sample_predictions.json'
    if samples_path.exists():
        with open(samples_path, 'r') as f:
            return json.load(f)
    
    # Return default samples if file doesn't exist
    return [
        {
            'input': get_sample_input(),
            'output': {
                'carbon_footprint_kg': 10.25,
                'carbon_impact_level': 'High',
                'confidence': 0.85
            }
        },
        {
            'input': {
                'day_type': 'Weekend',
                'transport_mode': 'Bike',
                'distance_km': 5.0,
                'electricity_kwh': 3.5,
                'renewable_usage_pct': 75.0,
                'food_type': 'Veg',
                'screen_time_hours': 3.0,
                'waste_generated_kg': 0.3,
                'eco_actions': 5
            },
            'output': {
                'carbon_footprint_kg': 4.75,
                'carbon_impact_level': 'Low',
                'confidence': 0.92
            }
        }
    ]

# Package metadata
__version__ = "1.0.0"
__author__ = "Carbon Footprint Prediction Team"
__description__ = "Trained models for Carbon Footprint Prediction"

# Print initialization message
print(f"Carbon Footprint Models v{__version__}")
print(f"Models directory: {MODELS_DIR}")
print(f"Available models: {', '.join(list_available_models())}")