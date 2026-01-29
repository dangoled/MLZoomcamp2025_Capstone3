"""
Prediction functions for Carbon Footprint API
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pickle
from pathlib import Path
import json

from .schemas import CarbonFootprintInput, CarbonFootprintResponse, PredictionExplanation
from src.preprocess import CarbonFootprintPreprocessor, ModelLoader

class CarbonFootprintPredictor:
    """
    Main predictor class for carbon footprint predictions
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize predictor with models from directory
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.preprocessor = None
        self.label_encoder = None
        self.metrics = {}
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all models from models directory"""
        try:
            # Load preprocessor
            preprocessor_path = self.models_dir / 'preprocessor.pkl'
            if preprocessor_path.exists():
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
            
            # Load label encoder
            label_encoder_path = self.models_dir / 'label_encoder.pkl'
            if label_encoder_path.exists():
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            
            # Load regression models
            regression_files = list(self.models_dir.glob('regression_*.pkl'))
            for file in regression_files:
                model_name = file.stem.replace('regression_', '')
                with open(file, 'rb') as f:
                    self.models[f'regression_{model_name}'] = pickle.load(f)
            
            # Load classification models
            classification_files = list(self.models_dir.glob('classification_*.pkl'))
            for file in classification_files:
                model_name = file.stem.replace('classification_', '')
                with open(file, 'rb') as f:
                    self.models[f'classification_{model_name}'] = pickle.load(f)
            
            # Load best models
            best_reg_path = self.models_dir / 'best_regression_model.pkl'
            if best_reg_path.exists():
                with open(best_reg_path, 'rb') as f:
                    self.models['best_regression'] = pickle.load(f)
            
            best_clf_path = self.models_dir / 'best_classification_model.pkl'
            if best_clf_path.exists():
                with open(best_clf_path, 'rb') as f:
                    self.models['best_classification'] = pickle.load(f)
            
            # Load all models
            all_models_path = self.models_dir / 'all_models.pkl'
            if all_models_path.exists():
                with open(all_models_path, 'rb') as f:
                    all_models = pickle.load(f)
                    self.models.update(all_models)
            
            # Load metrics
            metrics_path = self.models_dir / 'training_results.json'
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
            
            print(f"Loaded {len(self.models)} models from {self.models_dir}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def predict(self, input_data: Dict[str, Any], 
                model_type: str = 'best') -> Dict[str, Any]:
        """
        Predict carbon footprint for input data
        
        Args:
            input_data: Dictionary containing input features
            model_type: Type of model to use ('best', 'regression', 'classification', 
                       or specific model name)
        
        Returns:
            Dictionary with predictions and additional information
        """
        # Validate input
        self._validate_input(input_data)
        
        # Convert to DataFrame
        input_df = self._prepare_input(input_data)
        
        # Get predictions
        predictions = self._make_predictions(input_df, model_type)
        
        # Add suggestions
        predictions['suggestions'] = self._generate_suggestions(
            input_data, predictions['carbon_impact_level']
        )
        
        # Add feature contributions
        predictions['feature_contributions'] = self._get_feature_contributions(input_df)
        
        return predictions
    
    def _validate_input(self, input_data: Dict[str, Any]):
        """Validate input data"""
        required_fields = [
            'day_type', 'transport_mode', 'distance_km', 
            'electricity_kwh', 'renewable_usage_pct', 'food_type',
            'screen_time_hours', 'waste_generated_kg', 'eco_actions'
        ]
        
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate categorical values
        valid_day_types = ['Weekday', 'Weekend']
        if input_data['day_type'] not in valid_day_types:
            raise ValueError(f"day_type must be one of: {valid_day_types}")
        
        valid_transport_modes = ['Walk', 'Bike', 'Bus', 'EV', 'Car']
        if input_data['transport_mode'] not in valid_transport_modes:
            raise ValueError(f"transport_mode must be one of: {valid_transport_modes}")
        
        valid_food_types = ['Veg', 'Mixed', 'Non-Veg']
        if input_data['food_type'] not in valid_food_types:
            raise ValueError(f"food_type must be one of: {valid_food_types}")
        
        # Validate numeric ranges
        if input_data['distance_km'] < 0 or input_data['distance_km'] > 100:
            raise ValueError("distance_km must be between 0 and 100")
        
        if input_data['electricity_kwh'] < 0 or input_data['electricity_kwh'] > 50:
            raise ValueError("electricity_kwh must be between 0 and 50")
        
        if input_data['renewable_usage_pct'] < 0 or input_data['renewable_usage_pct'] > 100:
            raise ValueError("renewable_usage_pct must be between 0 and 100")
        
        if input_data['screen_time_hours'] < 0 or input_data['screen_time_hours'] > 24:
            raise ValueError("screen_time_hours must be between 0 and 24")
        
        if input_data['waste_generated_kg'] < 0 or input_data['waste_generated_kg'] > 10:
            raise ValueError("waste_generated_kg must be between 0 and 10")
        
        if input_data['eco_actions'] < 0 or input_data['eco_actions'] > 10:
            raise ValueError("eco_actions must be between 0 and 10")
    
    def _prepare_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare input data for prediction"""
        # Create DataFrame in correct order
        expected_columns = [
            'day_type', 'transport_mode', 'distance_km', 
            'electricity_kwh', 'renewable_usage_pct', 'food_type',
            'screen_time_hours', 'waste_generated_kg', 'eco_actions'
        ]
        
        df_dict = {col: [input_data[col]] for col in expected_columns}
        df = pd.DataFrame(df_dict)
        
        return df
    
    def _make_predictions(self, input_df: pd.DataFrame, 
                         model_type: str) -> Dict[str, Any]:
        """
        Make predictions using specified model type
        """
        predictions = {}
        
        # Get regression prediction
        if model_type == 'best':
            regression_model = self.models.get('best_regression')
            if regression_model is None:
                # Fallback to first regression model
                regression_models = [k for k in self.models.keys() if k.startswith('regression_')]
                if regression_models:
                    regression_model = self.models[regression_models[0]]
                else:
                    raise ValueError("No regression models available")
        elif model_type.startswith('regression_'):
            regression_model = self.models.get(model_type)
            if regression_model is None:
                raise ValueError(f"Model {model_type} not found")
        else:
            # Find any regression model
            regression_models = [k for k in self.models.keys() if k.startswith('regression_')]
            if regression_models:
                regression_model = self.models[regression_models[0]]
            else:
                raise ValueError("No regression models available")
        
        # Get classification prediction
        if model_type == 'best':
            classification_model = self.models.get('best_classification')
            if classification_model is None:
                # Fallback to first classification model
                classification_models = [k for k in self.models.keys() if k.startswith('classification_')]
                if classification_models:
                    classification_model = self.models[classification_models[0]]
                else:
                    raise ValueError("No classification models available")
        elif model_type.startswith('classification_'):
            classification_model = self.models.get(model_type)
            if classification_model is None:
                raise ValueError(f"Model {model_type} not found")
        else:
            # Find any classification model
            classification_models = [k for k in self.models.keys() if k.startswith('classification_')]
            if classification_models:
                classification_model = self.models[classification_models[0]]
            else:
                raise ValueError("No classification models available")
        
        # Make regression prediction
        try:
            carbon_footprint = regression_model.predict(input_df)[0]
            predictions['carbon_footprint_kg'] = float(carbon_footprint)
            
            # Get prediction confidence if available
            if hasattr(regression_model, 'predict_proba'):
                # For regression, we can estimate confidence from ensemble models
                predictions['confidence'] = 0.8  # Placeholder
            else:
                predictions['confidence'] = 0.7
        
        except Exception as e:
            raise ValueError(f"Regression prediction failed: {e}")
        
        # Make classification prediction
        try:
            impact_level_encoded = classification_model.predict(input_df)[0]
            
            # Decode label if encoder available
            if self.label_encoder:
                impact_level = self.label_encoder.inverse_transform([impact_level_encoded])[0]
            else:
                # Fallback: assume encoding 0=Low, 1=Medium, 2=High
                level_map = {0: 'Low', 1: 'Medium', 2: 'High'}
                impact_level = level_map.get(int(impact_level_encoded), 'Medium')
            
            predictions['carbon_impact_level'] = impact_level
            
            # Get probabilities if available
            if hasattr(classification_model, 'predict_proba'):
                probabilities = classification_model.predict_proba(input_df)[0]
                if self.label_encoder:
                    class_names = self.label_encoder.classes_
                    predictions['probabilities'] = {
                        cls: float(prob) for cls, prob in zip(class_names, probabilities)
                    }
                else:
                    predictions['probabilities'] = {
                        'Low': float(probabilities[0]),
                        'Medium': float(probabilities[1]),
                        'High': float(probabilities[2])
                    }
                
                # Confidence is max probability
                predictions['confidence'] = float(np.max(probabilities))
            else:
                predictions['confidence'] = 0.7
        
        except Exception as e:
            raise ValueError(f"Classification prediction failed: {e}")
        
        return predictions
    
    def _generate_suggestions(self, input_data: Dict[str, Any], 
                            impact_level: str) -> List[Dict[str, str]]:
        """
        Generate personalized suggestions to reduce carbon footprint
        """
        suggestions = []
        
        # High impact suggestions
        if impact_level == 'High':
            if input_data['transport_mode'] == 'Car' and input_data['distance_km'] > 5:
                suggestions.append({
                    'category': 'Transport',
                    'suggestion': 'Consider using public transport, cycling, or walking for trips under 10km',
                    'potential_reduction': '1-3 kg CO₂/day'
                })
            
            if input_data['renewable_usage_pct'] < 50:
                suggestions.append({
                    'category': 'Energy',
                    'suggestion': 'Switch to a renewable energy provider or consider solar panels',
                    'potential_reduction': '0.5-2 kg CO₂/day'
                })
            
            if input_data['food_type'] == 'Non-Veg':
                suggestions.append({
                    'category': 'Food',
                    'suggestion': 'Try incorporating more plant-based meals into your diet',
                    'potential_reduction': '0.5-1.5 kg CO₂/day'
                })
        
        # Medium impact suggestions
        elif impact_level == 'Medium':
            if input_data['electricity_kwh'] > 8:
                suggestions.append({
                    'category': 'Energy',
                    'suggestion': 'Reduce electricity usage by turning off unused appliances and using LED lights',
                    'potential_reduction': '0.3-1 kg CO₂/day'
                })
            
            if input_data['waste_generated_kg'] > 0.7:
                suggestions.append({
                    'category': 'Waste',
                    'suggestion': 'Improve recycling and composting to reduce waste',
                    'potential_reduction': '0.2-0.5 kg CO₂/day'
                })
            
            if input_data['eco_actions'] < 3:
                suggestions.append({
                    'category': 'Lifestyle',
                    'suggestion': 'Increase eco-friendly actions like using reusable bags and bottles',
                    'potential_reduction': '0.1-0.3 kg CO₂/day'
                })
        
        # Low impact suggestions (maintenance)
        else:
            suggestions.append({
                'category': 'Maintenance',
                'suggestion': 'Great job! Continue your sustainable practices',
                'potential_reduction': 'Keep up the good work!'
            })
            
            if input_data['renewable_usage_pct'] < 100:
                suggestions.append({
                    'category': 'Energy',
                    'suggestion': 'Consider reaching 100% renewable energy',
                    'potential_reduction': 'Additional 0.2-0.5 kg CO₂/day reduction'
                })
        
        # Add general suggestions based on specific features
        if input_data['screen_time_hours'] > 6:
            suggestions.append({
                'category': 'Digital',
                'suggestion': 'Reduce screen time and enable energy saving modes on devices',
                'potential_reduction': '0.1-0.3 kg CO₂/day'
            })
        
        # Limit to top 3 suggestions
        return suggestions[:3]
    
    def _get_feature_contributions(self, input_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate feature contributions to the prediction
        Simplified version - in production would use SHAP or similar
        """
        contributions = {}
        
        try:
            # This is a simplified calculation
            # In production, you would use SHAP values or model-specific methods
            
            # Example calculation based on feature values and typical impact
            feature_weights = {
                'transport_mode': {'Walk': 0.1, 'Bike': 0.2, 'Bus': 0.5, 'EV': 0.7, 'Car': 1.0},
                'distance_km': 0.05,  # per km
                'electricity_kwh': 0.3,  # per kWh
                'renewable_usage_pct': -0.02,  # per percent (negative reduces footprint)
                'food_type': {'Veg': 0.2, 'Mixed': 0.5, 'Non-Veg': 0.8},
                'screen_time_hours': 0.1,
                'waste_generated_kg': 0.4,
                'eco_actions': -0.15,  # per action (negative reduces footprint)
                'day_type': {'Weekday': 0, 'Weekend': -0.1}  # weekends often lower
            }
            
            for feature in input_df.columns:
                value = input_df[feature].iloc[0]
                
                if feature in feature_weights:
                    if isinstance(feature_weights[feature], dict):
                        # Categorical feature
                        weight = feature_weights[feature].get(str(value), 0)
                    else:
                        # Numerical feature
                        weight = feature_weights[feature] * float(value)
                    
                    contributions[feature] = float(weight)
        
        except Exception as e:
            # If calculation fails, return empty dict
            print(f"Feature contribution calculation failed: {e}")
        
        return contributions
    
    def predict_batch(self, input_data_list: List[Dict[str, Any]], 
                     model_type: str = 'best') -> List[Dict[str, Any]]:
        """
        Predict carbon footprint for multiple inputs
        
        Args:
            input_data_list: List of input dictionaries
            model_type: Type of model to use
        
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for i, input_data in enumerate(input_data_list):
            try:
                pred = self.predict(input_data, model_type)
                predictions.append(pred)
            except Exception as e:
                # Log error but continue with other predictions
                print(f"Error predicting sample {i}: {e}")
                predictions.append({
                    'error': str(e),
                    'carbon_footprint_kg': None,
                    'carbon_impact_level': 'Error',
                    'confidence': 0.0,
                    'suggestions': []
                })
        
        return predictions
    
    def explain_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide detailed explanation for a prediction
        
        Args:
            input_data: Input data dictionary
        
        Returns:
            Dictionary with prediction explanation
        """
        # Get prediction
        prediction = self.predict(input_data)
        
        # Get feature contributions
        input_df = self._prepare_input(input_data)
        feature_contributions = self._get_feature_contributions(input_df)
        
        # Calculate percentage contributions
        total_impact = sum(abs(v) for v in feature_contributions.values())
        if total_impact > 0:
            percentage_contributions = {
                k: float(abs(v) / total_impact * 100)
                for k, v in feature_contributions.items()
            }
        else:
            percentage_contributions = {}
        
        # Sort by contribution
        sorted_contributions = sorted(
            percentage_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            input_data, prediction, sorted_contributions
        )
        
        return {
            'prediction': prediction,
            'feature_contributions': feature_contributions,
            'percentage_contributions': dict(sorted_contributions[:5]),  # Top 5
            'explanation': explanation_text,
            'key_factors': self._get_key_factors(sorted_contributions)
        }
    
    def _generate_explanation_text(self, input_data: Dict[str, Any],
                                  prediction: Dict[str, Any],
                                  contributions: List[Tuple[str, float]]) -> str:
        """Generate human-readable explanation text"""
        footprint = prediction['carbon_footprint_kg']
        impact_level = prediction['carbon_impact_level']
        
        explanation = f"Your predicted carbon footprint is {footprint:.1f} kg CO₂ per day, "
        explanation += f"which is classified as {impact_level} impact.\n\n"
        
        if contributions:
            top_factor, top_percent = contributions[0]
            explanation += f"The main factor contributing to your footprint is "
            explanation += f"'{top_factor}' ({top_percent:.1f}%). "
            
            if len(contributions) > 1:
                second_factor, second_percent = contributions[1]
                explanation += f"Followed by '{second_factor}' ({second_percent:.1f}%)."
        
        # Add specific observations
        if input_data['transport_mode'] == 'Car' and input_data['distance_km'] > 10:
            explanation += "\n\nNote: Using a car for longer distances significantly increases your carbon footprint."
        
        if input_data['renewable_usage_pct'] < 30:
            explanation += "\n\nNote: Low renewable energy usage contributes to higher emissions."
        
        if input_data['food_type'] == 'Non-Veg' and impact_level in ['Medium', 'High']:
            explanation += "\n\nNote: Non-vegetarian diets typically have higher carbon footprints."
        
        return explanation
    
    def _get_key_factors(self, contributions: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """Get key factors from contributions"""
        key_factors = []
        
        for factor, percentage in contributions[:3]:  # Top 3
            factor_info = {
                'factor': factor,
                'percentage': percentage,
                'description': self._get_factor_description(factor)
            }
            key_factors.append(factor_info)
        
        return key_factors
    
    def _get_factor_description(self, factor: str) -> str:
        """Get description for a factor"""
        descriptions = {
            'transport_mode': 'Mode of transportation used',
            'distance_km': 'Distance traveled',
            'electricity_kwh': 'Electricity consumption',
            'renewable_usage_pct': 'Percentage of renewable energy used',
            'food_type': 'Type of diet',
            'screen_time_hours': 'Screen time duration',
            'waste_generated_kg': 'Amount of waste generated',
            'eco_actions': 'Number of eco-friendly actions taken',
            'day_type': 'Day type (Weekday/Weekend)'
        }
        
        return descriptions.get(factor, factor)
    
    def get_model_info(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about loaded models
        
        Args:
            model_type: Optional filter for model type
        
        Returns:
            Dictionary with model information
        """
        model_info = {
            'total_models': len(self.models),
            'models_loaded': True if self.models else False,
            'preprocessor_loaded': self.preprocessor is not None,
            'label_encoder_loaded': self.label_encoder is not None,
            'available_models': list(self.models.keys())
        }
        
        # Add performance metrics if available
        if self.metrics:
            model_info['performance_metrics'] = self.metrics
        
        # Filter by model type if specified
        if model_type:
            filtered_models = {
                k: v for k, v in self.models.items() 
                if model_type.lower() in k.lower()
            }
            model_info['filtered_models'] = list(filtered_models.keys())
            model_info['filtered_count'] = len(filtered_models)
        
        # Add best model info
        if 'best_regression' in self.models:
            model_info['best_regression_model'] = 'best_regression'
        
        if 'best_classification' in self.models:
            model_info['best_classification_model'] = 'best_classification'
        
        return model_info
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models by type"""
        regression_models = [m for m in self.models.keys() if m.startswith('regression_')]
        classification_models = [m for m in self.models.keys() if m.startswith('classification_')]
        
        return {
            'regression': regression_models,
            'classification': classification_models,
            'best': ['best_regression', 'best_classification']
        }
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about predictions made
        
        Returns:
            Dictionary with prediction statistics
        """
        # This would typically track actual prediction statistics
        # For now, return placeholder statistics
        return {
            'average_footprint': 8.08,  # From dataset mean
            'footprint_range': {'min': 1.79, 'max': 16.02},
            'impact_level_distribution': {
                'Low': 27.6,
                'Medium': 49.4,
                'High': 23.0
            },
            'most_common_transport': 'Car',
            'average_renewable_usage': 35.2
        }

# ============================================================================
# BATCH PREDICTION UTILITIES
# ============================================================================

class BatchPredictor:
    """
    Utility class for batch predictions
    """
    
    def __init__(self, predictor: CarbonFootprintPredictor):
        self.predictor = predictor
    
    def predict_from_csv(self, csv_path: str, 
                        output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Predict from CSV file
        
        Args:
            csv_path: Path to input CSV file
            output_path: Optional path to save results
        
        Returns:
            DataFrame with predictions
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Validate columns
        required_columns = [
            'day_type', 'transport_mode', 'distance_km', 
            'electricity_kwh', 'renewable_usage_pct', 'food_type',
            'screen_time_hours', 'waste_generated_kg', 'eco_actions'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")
        
        # Make predictions for each row
        predictions = []
        
        for idx, row in df.iterrows():
            input_data = row[required_columns].to_dict()
            
            try:
                pred = self.predictor.predict(input_data)
                
                # Add original data and prediction
                result = {**input_data, **pred}
                predictions.append(result)
                
            except Exception as e:
                print(f"Error predicting row {idx}: {e}")
                # Add error information
                result = {**input_data, 'error': str(e)}
                predictions.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(predictions)
        
        # Save to file if output path specified
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        return results_df
    
    def analyze_batch_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze batch prediction results
        
        Args:
            results_df: DataFrame with prediction results
        
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis['total_samples'] = len(results_df)
        
        if 'carbon_footprint_kg' in results_df.columns:
            analysis['footprint_stats'] = {
                'mean': float(results_df['carbon_footprint_kg'].mean()),
                'median': float(results_df['carbon_footprint_kg'].median()),
                'std': float(results_df['carbon_footprint_kg'].std()),
                'min': float(results_df['carbon_footprint_kg'].min()),
                'max': float(results_df['carbon_footprint_kg'].max())
            }
        
        if 'carbon_impact_level' in results_df.columns:
            impact_counts = results_df['carbon_impact_level'].value_counts()
            analysis['impact_distribution'] = impact_counts.to_dict()
            
            # Calculate percentages
            total = len(results_df)
            analysis['impact_percentages'] = {
                level: float(count / total * 100)
                for level, count in impact_counts.items()
            }
        
        # Error analysis
        if 'error' in results_df.columns:
            error_count = results_df['error'].notna().sum()
            analysis['error_count'] = int(error_count)
            analysis['error_rate'] = float(error_count / len(results_df) * 100)
        
        # Correlation analysis
        numeric_columns = ['distance_km', 'electricity_kwh', 'renewable_usage_pct',
                          'screen_time_hours', 'waste_generated_kg', 'eco_actions',
                          'carbon_footprint_kg']
        
        numeric_df = results_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        correlation_matrix = numeric_df.corr()
        
        if 'carbon_footprint_kg' in correlation_matrix.columns:
            footprint_corr = correlation_matrix['carbon_footprint_kg'].sort_values(
                key=abs, ascending=False
            )
            analysis['footprint_correlations'] = footprint_corr.to_dict()
        
        return analysis

# ============================================================================
# PREDICTION CACHE
# ============================================================================

class PredictionCache:
    """
    Simple cache for predictions to avoid redundant computations
    """
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get_key(self, input_data: Dict[str, Any]) -> str:
        """Generate cache key from input data"""
        # Sort items to ensure consistent keys
        sorted_items = sorted(input_data.items())
        return str(sorted_items)
    
    def get(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get prediction from cache"""
        key = self.get_key(input_data)
        
        if key in self.cache:
            # Update access count
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        
        return None
    
    def set(self, input_data: Dict[str, Any], prediction: Dict[str, Any]):
        """Store prediction in cache"""
        key = self.get_key(input_data)
        
        # Check cache size
        if len(self.cache) >= self.max_size:
            # Remove least accessed item
            self._evict_least_accessed()
        
        self.cache[key] = prediction
        self.access_count[key] = 1
    
    def _evict_least_accessed(self):
        """Evict least accessed item from cache"""
        if not self.access_count:
            # Remove random item if no access counts
            random_key = next(iter(self.cache.keys()))
            del self.cache[random_key]
            return
        
        # Find item with lowest access count
        min_key = min(self.access_count.items(), key=lambda x: x[1])[0]
        
        # Remove from both dictionaries
        del self.cache[min_key]
        del self.access_count[min_key]
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_count.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': self._calculate_hit_rate(),
            'total_accesses': sum(self.access_count.values())
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_hits = sum(self.access_count.values())
        total_accesses = total_hits + len(self.cache)  # Approximate
        return total_hits / total_accesses if total_accesses > 0 else 0.0

# ============================================================================
# PREDICTION VALIDATION
# ============================================================================

class PredictionValidator:
    """
    Validate predictions for consistency and plausibility
    """
    
    def __init__(self):
        self.footprint_bounds = (1.0, 20.0)  # Reasonable bounds in kg
        self.typical_values = {
            'transport_mode': ['Walk', 'Bike', 'Bus', 'EV', 'Car'],
            'food_type': ['Veg', 'Mixed', 'Non-Veg'],
            'day_type': ['Weekday', 'Weekend']
        }
    
    def validate_prediction(self, input_data: Dict[str, Any], 
                          prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a prediction
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check carbon footprint bounds
        footprint = prediction.get('carbon_footprint_kg')
        if footprint is not None:
            if not (self.footprint_bounds[0] <= footprint <= self.footprint_bounds[1]):
                validation_results['warnings'].append(
                    f"Carbon footprint ({footprint:.2f} kg) outside typical range "
                    f"({self.footprint_bounds[0]}-{self.footprint_bounds[1]} kg)"
                )
        
        # Check impact level consistency
        footprint = prediction.get('carbon_footprint_kg')
        impact_level = prediction.get('carbon_impact_level')
        
        if footprint is not None and impact_level is not None:
            expected_level = self._get_expected_impact_level(footprint)
            if expected_level != impact_level:
                validation_results['warnings'].append(
                    f"Impact level '{impact_level}' doesn't match expected level "
                    f"'{expected_level}' for footprint {footprint:.2f} kg"
                )
        
        # Check input consistency
        self._check_input_consistency(input_data, validation_results)
        
        # Check for extreme values
        self._check_extreme_values(input_data, validation_results)
        
        # Update is_valid based on errors
        if validation_results['errors']:
            validation_results['is_valid'] = False
        
        return validation_results
    
    def _get_expected_impact_level(self, footprint: float) -> str:
        """Get expected impact level based on footprint"""
        if footprint < 6.5:
            return 'Low'
        elif footprint <= 10.5:
            return 'Medium'
        else:
            return 'High'
    
    def _check_input_consistency(self, input_data: Dict[str, Any], 
                                validation_results: Dict[str, Any]):
        """Check for input consistency issues"""
        # Check if transport mode and distance are consistent
        transport = input_data.get('transport_mode')
        distance = input_data.get('distance_km')
        
        if transport == 'Walk' and distance > 30:
            validation_results['warnings'].append(
                f"Walking distance ({distance} km) seems unusually high"
            )
        
        if transport == 'Bike' and distance > 50:
            validation_results['warnings'].append(
                f"Biking distance ({distance} km) seems unusually high"
            )
        
        # Check renewable usage
        renewable = input_data.get('renewable_usage_pct', 0)
        if renewable > 100:
            validation_results['errors'].append(
                f"Renewable usage ({renewable}%) cannot exceed 100%"
            )
        
        # Check eco actions
        eco_actions = input_data.get('eco_actions', 0)
        if eco_actions < 0:
            validation_results['errors'].append(
                f"Eco actions ({eco_actions}) cannot be negative"
            )
    
    def _check_extreme_values(self, input_data: Dict[str, Any], 
                             validation_results: Dict[str, Any]):
        """Check for extreme or implausible values"""
        extreme_thresholds = {
            'distance_km': 100,
            'electricity_kwh': 30,
            'screen_time_hours': 18,
            'waste_generated_kg': 5,
            'eco_actions': 10
        }
        
        for feature, threshold in extreme_thresholds.items():
            value = input_data.get(feature)
            if value is not None and value > threshold:
                validation_results['warnings'].append(
                    f"{feature} ({value}) seems unusually high"
                )

# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================

def make_prediction(input_data: Dict[str, Any], 
                   models_dir: str = "models",
                   use_cache: bool = True) -> Dict[str, Any]:
    """
    Main function to make a prediction (convenience wrapper)
    
    Args:
        input_data: Input data dictionary
        models_dir: Directory containing models
        use_cache: Whether to use prediction cache
    
    Returns:
        Prediction dictionary
    """
    # Initialize predictor
    predictor = CarbonFootprintPredictor(models_dir)
    
    # Initialize cache if enabled
    cache = None
    if use_cache:
        cache = PredictionCache()
        cached_prediction = cache.get(input_data)
        if cached_prediction:
            cached_prediction['from_cache'] = True
            return cached_prediction
    
    # Make prediction
    prediction = predictor.predict(input_data)
    
    # Validate prediction
    validator = PredictionValidator()
    validation = validator.validate_prediction(input_data, prediction)
    prediction['validation'] = validation
    
    # Add timestamp
    prediction['timestamp'] = datetime.now().isoformat()
    
    # Cache the prediction
    if cache and validation['is_valid']:
        cache.set(input_data, prediction)
    
    return prediction

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    sample_input = {
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
    
    print("Making prediction for sample input...")
    
    try:
        prediction = make_prediction(sample_input)
        
        print("\nPrediction Results:")
        print(f"Carbon Footprint: {prediction['carbon_footprint_kg']:.2f} kg CO₂/day")
        print(f"Impact Level: {prediction['carbon_impact_level']}")
        print(f"Confidence: {prediction.get('confidence', 0.0):.2%}")
        
        print("\nSuggestions:")
        for suggestion in prediction.get('suggestions', []):
            print(f"- {suggestion['suggestion']} ({suggestion['potential_reduction']})")
        
        print("\nFeature Contributions:")
        for feature, contribution in prediction.get('feature_contributions', {}).items():
            print(f"- {feature}: {contribution:.3f}")
        
        if prediction.get('from_cache'):
            print("\nNote: Prediction served from cache")
    
    except Exception as e:
        print(f"Error making prediction: {e}")