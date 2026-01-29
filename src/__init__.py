"""
Carbon Footprint Prediction - Source Module
"""

from .preprocess import CarbonFootprintPreprocessor, ModelLoader
from .predict import (
    ModelPredictor, 
    load_predictor, 
    make_quick_prediction, 
    validate_and_predict
)

# Version information
__version__ = "1.0.0"
__author__ = "Carbon Footprint Prediction Team"
__description__ = "Personal Carbon Footprint Prediction System"

# Export main classes and functions
__all__ = [
    # Preprocessing
    "CarbonFootprintPreprocessor",
    "ModelLoader",
    
    # Prediction
    "ModelPredictor",
    "load_predictor",
    "make_quick_prediction",
    "validate_and_predict",
]

# Optional: Lazy loading for large modules
def __getattr__(name):
    """Lazy import for optional modules"""
    if name == "train":
        from . import train
        return train
    elif name == "model":
        from . import model
        return model
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Package initialization
print(f"Carbon Footprint Prediction v{__version__}")
print(f"Description: {__description__}")
print("Modules loaded: preprocess, predict")