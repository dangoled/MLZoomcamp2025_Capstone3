"""
Data preprocessing utilities for Carbon Footprint Prediction
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
import pickle
from pathlib import Path

class CarbonFootprintPreprocessor:
    """Preprocessor for carbon footprint data"""
    
    def __init__(self, preprocessor_path: str = None):
        """Initialize preprocessor"""
        self.preprocessor = None
        self.feature_names = None
        
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)
    
    def load_preprocessor(self, path: str):
        """Load preprocessor from file"""
        with open(path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        # Get feature names
        self._extract_feature_names()
    
    def _extract_feature_names(self):
        """Extract feature names from preprocessor"""
        if self.preprocessor is None:
            return
        
        # Get numeric features
        numeric_features = self.preprocessor.transformers_[0][2]
        
        # Get categorical features
        categorical_encoder = self.preprocessor.transformers_[1][1].named_steps['onehot']
        categorical_features = categorical_encoder.get_feature_names_out(
            self.preprocessor.transformers_[1][2]
        )
        
        # Combine all features
        self.feature_names = list(numeric_features) + list(categorical_features)
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform input data"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded. Call load_preprocessor() first.")
        
        return self.preprocessor.transform(df)
    
    def transform_with_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform input data and return as DataFrame with feature names"""
        transformed = self.transform(df)
        return pd.DataFrame(transformed, columns=self.feature_names)
    
    @staticmethod
    def prepare_input_dict(data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare input dictionary for prediction"""
        # Convert to DataFrame with correct column order
        expected_columns = ['day_type', 'transport_mode', 'distance_km', 
                          'electricity_kwh', 'renewable_usage_pct', 'food_type',
                          'screen_time_hours', 'waste_generated_kg', 'eco_actions']
        
        # Create DataFrame with all expected columns
        df_dict = {col: [data.get(col, None)] for col in expected_columns}
        df = pd.DataFrame(df_dict)
        
        # Convert numeric columns
        numeric_cols = ['distance_km', 'electricity_kwh', 'renewable_usage_pct',
                       'screen_time_hours', 'waste_generated_kg', 'eco_actions']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    @staticmethod
    def get_sample_input() -> Dict[str, Any]:
        """Get sample input for API testing"""
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

class ModelLoader:
    """Utility class for loading models"""
    
    @staticmethod
    def load_model(model_path: str):
        """Load model from file"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def load_all_models(models_dir: str = 'models'):
        """Load all models from directory"""
        models_dir = Path(models_dir)
        models = {}
        
        # Load regression models
        regression_files = list(models_dir.glob('regression_*.pkl'))
        for file in regression_files:
            model_name = file.stem.replace('regression_', '')
            models[f'regression_{model_name}'] = ModelLoader.load_model(file)
        
        # Load classification models
        classification_files = list(models_dir.glob('classification_*.pkl'))
        for file in classification_files:
            model_name = file.stem.replace('classification_', '')
            models[f'classification_{model_name}'] = ModelLoader.load_model(file)
        
        # Load preprocessor
        preprocessor_path = models_dir / 'preprocessor.pkl'
        if preprocessor_path.exists():
            models['preprocessor'] = ModelLoader.load_model(preprocessor_path)
        
        # Load metrics
        metrics_path = models_dir / 'metrics.json'
        if metrics_path.exists():
            import json
            with open(metrics_path, 'r') as f:
                models['metrics'] = json.load(f)
        
        return models