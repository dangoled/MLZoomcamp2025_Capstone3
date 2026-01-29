"""
Prediction utilities for Carbon Footprint models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import pickle
from pathlib import Path
import json
from datetime import datetime

class ModelPredictor:
    """
    Predictor class for making predictions with trained models
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
                print(f"✓ Loaded preprocessor from {preprocessor_path}")
            
            # Load label encoder
            label_encoder_path = self.models_dir / 'label_encoder.pkl'
            if label_encoder_path.exists():
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print(f"✓ Loaded label encoder from {label_encoder_path}")
            
            # Load best regression model
            best_reg_path = self.models_dir / 'best_regression_model.pkl'
            if best_reg_path.exists():
                with open(best_reg_path, 'rb') as f:
                    self.models['best_regression'] = pickle.load(f)
                print(f"✓ Loaded best regression model from {best_reg_path}")
            
            # Load best classification model
            best_clf_path = self.models_dir / 'best_classification_model.pkl'
            if best_clf_path.exists():
                with open(best_clf_path, 'rb') as f:
                    self.models['best_classification'] = pickle.load(f)
                print(f"✓ Loaded best classification model from {best_clf_path}")
            
            # Load all models (optional)
            all_models_path = self.models_dir / 'all_models.pkl'
            if all_models_path.exists():
                with open(all_models_path, 'rb') as f:
                    all_models_data = pickle.load(f)
                    # Extract models from the saved data structure
                    if 'regression' in all_models_data:
                        for name, model in all_models_data['regression'].items():
                            if 'model' in model:
                                self.models[f'regression_{name}'] = model['model']
                    if 'classification' in all_models_data:
                        for name, model in all_models_data['classification'].items():
                            if 'model' in model:
                                self.models[f'classification_{name}'] = model['model']
                    if 'preprocessor' in all_models_data:
                        self.preprocessor = all_models_data['preprocessor']
                    if 'label_encoder' in all_models_data:
                        self.label_encoder = all_models_data['label_encoder']
                print(f"✓ Loaded all models from {all_models_path}")
            
            # Load metrics
            metrics_path = self.models_dir / 'training_results.json'
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
                print(f"✓ Loaded metrics from {metrics_path}")
            
            print(f"Total models loaded: {len(self.models)}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate input data
        
        Args:
            input_data: Dictionary containing input features
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        required_fields = [
            'day_type', 'transport_mode', 'distance_km', 
            'electricity_kwh', 'renewable_usage_pct', 'food_type',
            'screen_time_hours', 'waste_generated_kg', 'eco_actions'
        ]
        
        errors = []
        
        # Check required fields
        for field in required_fields:
            if field not in input_data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Validate categorical values
        valid_day_types = ['Weekday', 'Weekend']
        if input_data['day_type'] not in valid_day_types:
            errors.append(f"day_type must be one of: {valid_day_types}")
        
        valid_transport_modes = ['Walk', 'Bike', 'Bus', 'EV', 'Car']
        if input_data['transport_mode'] not in valid_transport_modes:
            errors.append(f"transport_mode must be one of: {valid_transport_modes}")
        
        valid_food_types = ['Veg', 'Mixed', 'Non-Veg']
        if input_data['food_type'] not in valid_food_types:
            errors.append(f"food_type must be one of: {valid_food_types}")
        
        # Validate numeric ranges
        numeric_checks = [
            ('distance_km', 0, 100),
            ('electricity_kwh', 0, 50),
            ('renewable_usage_pct', 0, 100),
            ('screen_time_hours', 0, 24),
            ('waste_generated_kg', 0, 10),
            ('eco_actions', 0, 10)
        ]
        
        for field, min_val, max_val in numeric_checks:
            value = input_data[field]
            if not (min_val <= value <= max_val):
                errors.append(f"{field} must be between {min_val} and {max_val}")
        
        return len(errors) == 0, errors
    
    def prepare_input_dataframe(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert input dictionary to DataFrame
        
        Args:
            input_data: Dictionary containing input features
        
        Returns:
            DataFrame ready for prediction
        """
        # Create DataFrame in correct order
        expected_columns = [
            'day_type', 'transport_mode', 'distance_km', 
            'electricity_kwh', 'renewable_usage_pct', 'food_type',
            'screen_time_hours', 'waste_generated_kg', 'eco_actions'
        ]
        
        df_dict = {col: [input_data[col]] for col in expected_columns}
        df = pd.DataFrame(df_dict)
        
        return df
    
    def predict_regression(self, input_df: pd.DataFrame, 
                          model_name: str = 'best_regression') -> Tuple[float, float]:
        """
        Predict carbon footprint (regression)
        
        Args:
            input_df: Input DataFrame
            model_name: Name of regression model to use
        
        Returns:
            Tuple of (prediction, confidence)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        try:
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction = float(prediction)
            
            # Calculate confidence (simplified)
            # For ensemble models, could use prediction variance
            # For single models, could use distance from training data distribution
            confidence = self._calculate_regression_confidence(input_df, prediction, model_name)
            
            return prediction, confidence
            
        except Exception as e:
            raise ValueError(f"Regression prediction failed: {e}")
    
    def predict_classification(self, input_df: pd.DataFrame, 
                              model_name: str = 'best_classification') -> Tuple[str, float, Dict[str, float]]:
        """
        Predict impact level (classification)
        
        Args:
            input_df: Input DataFrame
            model_name: Name of classification model to use
        
        Returns:
            Tuple of (prediction, confidence, probabilities)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        try:
            # Make prediction
            impact_level_encoded = model.predict(input_df)[0]
            
            # Decode label if encoder available
            if self.label_encoder:
                impact_level = self.label_encoder.inverse_transform([impact_level_encoded])[0]
            else:
                # Fallback: assume encoding 0=Low, 1=Medium, 2=High
                level_map = {0: 'Low', 1: 'Medium', 2: 'High'}
                impact_level = level_map.get(int(impact_level_encoded), 'Medium')
            
            # Get probabilities if available
            probabilities = {}
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)[0]
                if self.label_encoder:
                    class_names = self.label_encoder.classes_
                    probabilities = {
                        str(cls): float(prob) for cls, prob in zip(class_names, proba)
                    }
                else:
                    probabilities = {
                        'Low': float(proba[0]),
                        'Medium': float(proba[1]),
                        'High': float(proba[2])
                    }
                
                # Confidence is max probability
                confidence = float(np.max(proba))
            else:
                confidence = 0.7
                probabilities = {
                    impact_level: 1.0
                }
            
            return str(impact_level), confidence, probabilities
            
        except Exception as e:
            raise ValueError(f"Classification prediction failed: {e}")
    
    def _calculate_regression_confidence(self, input_df: pd.DataFrame, 
                                        prediction: float, 
                                        model_name: str) -> float:
        """
        Calculate confidence score for regression prediction
        
        Args:
            input_df: Input DataFrame
            prediction: Predicted value
            model_name: Name of model used
        
        Returns:
            Confidence score between 0 and 1
        """
        # Simplified confidence calculation
        # In production, this could be based on:
        # 1. Distance from training data distribution
        # 2. Ensemble variance
        # 3. Model-specific uncertainty estimates
        
        # Get typical range from training data if available
        if self.metrics and 'dataset_info' in self.metrics:
            target_info = self.metrics['dataset_info'].get('target_distribution', {})
            footprint_info = target_info.get('carbon_footprint_kg', {})
            
            mean = footprint_info.get('mean', 8.08)
            std = footprint_info.get('std', 2.5)
            min_val = footprint_info.get('min', 1.79)
            max_val = footprint_info.get('max', 16.02)
            
            # Check if prediction is within typical range
            if min_val <= prediction <= max_val:
                # Calculate z-score (how many standard deviations from mean)
                z_score = abs(prediction - mean) / std
                
                # Convert to confidence (higher confidence for predictions closer to mean)
                # Using exponential decay: confidence = exp(-z_score^2 / 2)
                confidence = np.exp(-0.5 * z_score ** 2)
                
                # Ensure reasonable bounds
                confidence = max(0.5, min(0.95, confidence))
                
                return confidence
        
        # Default confidence
        return 0.7
    
    def predict(self, input_data: Dict[str, Any], 
                regression_model: str = 'best_regression',
                classification_model: str = 'best_classification') -> Dict[str, Any]:
        """
        Make complete prediction for input data
        
        Args:
            input_data: Dictionary containing input features
            regression_model: Name of regression model to use
            classification_model: Name of classification model to use
        
        Returns:
            Dictionary with predictions and additional information
        """
        # Validate input
        is_valid, errors = self.validate_input(input_data)
        if not is_valid:
            raise ValueError(f"Invalid input: {', '.join(errors)}")
        
        # Prepare input
        input_df = self.prepare_input_dataframe(input_data)
        
        # Make predictions
        try:
            carbon_footprint, reg_confidence = self.predict_regression(input_df, regression_model)
            impact_level, clf_confidence, probabilities = self.predict_classification(input_df, classification_model)
            
            # Calculate overall confidence (average of both)
            overall_confidence = (reg_confidence + clf_confidence) / 2
            
            # Generate suggestions
            suggestions = self._generate_suggestions(input_data, impact_level, carbon_footprint)
            
            # Get feature contributions
            feature_contributions = self._calculate_feature_contributions(input_df, input_data)
            
            # Prepare result
            result = {
                'carbon_footprint_kg': carbon_footprint,
                'carbon_impact_level': impact_level,
                'confidence': overall_confidence,
                'regression_confidence': reg_confidence,
                'classification_confidence': clf_confidence,
                'probabilities': probabilities,
                'suggestions': suggestions,
                'feature_contributions': feature_contributions,
                'timestamp': datetime.now().isoformat(),
                'input_summary': self._create_input_summary(input_data)
            }
            
            return result
            
        except Exception as e:
            raise ValueError(f"Prediction failed: {e}")
    
    def _generate_suggestions(self, input_data: Dict[str, Any], 
                             impact_level: str, 
                             carbon_footprint: float) -> List[Dict[str, str]]:
        """
        Generate personalized suggestions to reduce carbon footprint
        
        Args:
            input_data: Original input data
            impact_level: Predicted impact level
            carbon_footprint: Predicted carbon footprint
        
        Returns:
            List of suggestion dictionaries
        """
        suggestions = []
        
        # High impact suggestions
        if impact_level == 'High':
            if input_data['transport_mode'] == 'Car' and input_data['distance_km'] > 5:
                suggestions.append({
                    'category': 'Transport',
                    'action': 'Switch to public transport, cycling, or walking',
                    'impact': 'High',
                    'estimated_reduction': '1-3 kg CO₂/day',
                    'reason': 'Car usage is the largest contributor to your carbon footprint'
                })
            
            if input_data['renewable_usage_pct'] < 30:
                suggestions.append({
                    'category': 'Energy',
                    'action': 'Increase renewable energy usage',
                    'impact': 'High',
                    'estimated_reduction': '0.5-2 kg CO₂/day',
                    'reason': 'Low renewable energy percentage increases emissions'
                })
            
            if input_data['food_type'] == 'Non-Veg':
                suggestions.append({
                    'category': 'Food',
                    'action': 'Incorporate more plant-based meals',
                    'impact': 'Medium',
                    'estimated_reduction': '0.5-1.5 kg CO₂/day',
                    'reason': 'Non-vegetarian diets have higher carbon footprints'
                })
        
        # Medium impact suggestions
        elif impact_level == 'Medium':
            if input_data['electricity_kwh'] > 8:
                suggestions.append({
                    'category': 'Energy',
                    'action': 'Reduce electricity consumption',
                    'impact': 'Medium',
                    'estimated_reduction': '0.3-1 kg CO₂/day',
                    'reason': 'High electricity usage contributes significantly'
                })
            
            if input_data['waste_generated_kg'] > 0.7:
                suggestions.append({
                    'category': 'Waste',
                    'action': 'Improve recycling and composting',
                    'impact': 'Medium',
                    'estimated_reduction': '0.2-0.5 kg CO₂/day',
                    'reason': 'Reducing waste lowers emissions from landfills'
                })
            
            if input_data['eco_actions'] < 3:
                suggestions.append({
                    'category': 'Lifestyle',
                    'action': 'Increase eco-friendly actions',
                    'impact': 'Low',
                    'estimated_reduction': '0.1-0.3 kg CO₂/day',
                    'reason': 'More eco-actions can reduce your footprint'
                })
        
        # Low impact suggestions (maintenance and improvement)
        else:
            suggestions.append({
                'category': 'Maintenance',
                'action': 'Continue sustainable practices',
                'impact': 'N/A',
                'estimated_reduction': 'Maintain current level',
                'reason': 'You are already doing well! Keep up the good work'
            })
            
            if input_data['renewable_usage_pct'] < 100:
                suggestions.append({
                    'category': 'Energy',
                    'action': 'Aim for 100% renewable energy',
                    'impact': 'Low',
                    'estimated_reduction': 'Additional 0.2-0.5 kg CO₂/day',
                    'reason': 'Achieving full renewable usage further reduces emissions'
                })
        
        # Add general suggestions based on specific features
        if input_data['screen_time_hours'] > 6:
            suggestions.append({
                'category': 'Digital',
                'action': 'Reduce screen time and enable power saving',
                'impact': 'Low',
                'estimated_reduction': '0.1-0.3 kg CO₂/day',
                'reason': 'High screen time increases electricity consumption'
            })
        
        # Ensure we don't have too many suggestions
        return suggestions[:3]
    
    def _calculate_feature_contributions(self, input_df: pd.DataFrame, 
                                        input_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate approximate feature contributions to the prediction
        
        Args:
            input_df: Input DataFrame
            input_data: Original input data
        
        Returns:
            Dictionary of feature contributions
        """
        contributions = {}
        
        try:
            # Simplified feature contribution calculation
            # Based on domain knowledge and typical impact factors
            
            # Transport contribution
            transport_scores = {
                'Walk': 0.1, 'Bike': 0.2, 'Bus': 0.5, 'EV': 0.7, 'Car': 1.0
            }
            transport = input_data['transport_mode']
            distance = input_data['distance_km']
            contributions['transport'] = transport_scores.get(transport, 0.5) * (distance / 10)
            
            # Energy contribution
            electricity = input_data['electricity_kwh']
            renewable = input_data['renewable_usage_pct']
            # Non-renewable electricity has higher impact
            non_renewable_factor = (100 - renewable) / 100
            contributions['energy'] = electricity * non_renewable_factor * 0.3
            
            # Food contribution
            food_scores = {'Veg': 0.2, 'Mixed': 0.5, 'Non-Veg': 0.8}
            contributions['food'] = food_scores.get(input_data['food_type'], 0.5)
            
            # Waste contribution
            contributions['waste'] = input_data['waste_generated_kg'] * 0.4
            
            # Eco actions (negative contribution - reduces footprint)
            contributions['eco_actions'] = -input_data['eco_actions'] * 0.15
            
            # Screen time contribution
            contributions['screen_time'] = input_data['screen_time_hours'] * 0.1
            
            # Normalize contributions to sum to approximate footprint
            total = sum(abs(v) for v in contributions.values())
            if total > 0:
                # Scale to match typical footprint range
                scale_factor = 8.0 / total  # Scale to ~8 kg average
                contributions = {k: v * scale_factor for k, v in contributions.items()}
            
            # Round to 3 decimal places
            contributions = {k: round(v, 3) for k, v in contributions.items()}
            
        except Exception as e:
            print(f"Feature contribution calculation failed: {e}")
            contributions = {}
        
        return contributions
    
    def _create_input_summary(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of the input data
        
        Args:
            input_data: Original input data
        
        Returns:
            Summary dictionary
        """
        return {
            'transport': f"{input_data['transport_mode']} ({input_data['distance_km']} km)",
            'energy': f"{input_data['electricity_kwh']} kWh, {input_data['renewable_usage_pct']}% renewable",
            'food': input_data['food_type'],
            'lifestyle': {
                'screen_time': f"{input_data['screen_time_hours']} hours",
                'waste': f"{input_data['waste_generated_kg']} kg",
                'eco_actions': input_data['eco_actions']
            },
            'day_type': input_data['day_type']
        }
    
    def explain_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide detailed explanation for a prediction
        
        Args:
            input_data: Input data dictionary
        
        Returns:
            Dictionary with prediction explanation
        """
        # Get prediction first
        prediction = self.predict(input_data)
        
        # Get feature contributions
        feature_contributions = prediction.get('feature_contributions', {})
        
        # Calculate percentage contributions
        total_impact = sum(abs(v) for v in feature_contributions.values())
        if total_impact > 0:
            percentage_contributions = {
                k: round(abs(v) / total_impact * 100, 1)
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
        
        # Generate explanation
        explanation = self._generate_explanation(input_data, prediction, sorted_contributions)
        
        return {
            'prediction': {
                'carbon_footprint_kg': prediction['carbon_footprint_kg'],
                'carbon_impact_level': prediction['carbon_impact_level'],
                'confidence': prediction['confidence']
            },
            'feature_contributions': feature_contributions,
            'percentage_contributions': dict(sorted_contributions),
            'top_factors': [
                {'factor': factor, 'percentage': perc}
                for factor, perc in sorted_contributions[:3]
            ],
            'explanation': explanation,
            'suggestions': prediction['suggestions']
        }
    
    def _generate_explanation(self, input_data: Dict[str, Any],
                             prediction: Dict[str, Any],
                             contributions: List[Tuple[str, float]]) -> str:
        """
        Generate human-readable explanation
        
        Args:
            input_data: Input data
            prediction: Prediction results
            contributions: Feature contributions
        
        Returns:
            Explanation text
        """
        footprint = prediction['carbon_footprint_kg']
        impact_level = prediction['carbon_impact_level']
        
        # Start explanation
        explanation_parts = []
        
        # Main statement
        explanation_parts.append(
            f"Your daily carbon footprint is estimated at {footprint:.1f} kg CO₂, "
            f"which is classified as **{impact_level}** impact."
        )
        
        # Add context
        if impact_level == 'Low':
            explanation_parts.append(
                "This is below average and indicates sustainable lifestyle choices."
            )
        elif impact_level == 'Medium':
            explanation_parts.append(
                "This is around average. There are opportunities for improvement."
            )
        else:  # High
            explanation_parts.append(
                "This is above average. Significant reductions are possible with changes."
            )
        
        # Top factors
        if contributions:
            top_factor, top_percent = contributions[0]
            explanation_parts.append(
                f"\nThe main contributor to your footprint is **{top_factor}** "
                f"({top_percent:.1f}% of total impact)."
            )
            
            if len(contributions) > 1:
                second_factor, second_percent = contributions[1]
                explanation_parts.append(
                    f"Followed by **{second_factor}** ({second_percent:.1f}%)."
                )
        
        # Specific observations
        observations = []
        
        if input_data['transport_mode'] == 'Car' and input_data['distance_km'] > 10:
            observations.append("Using a car for longer distances significantly increases emissions.")
        
        if input_data['renewable_usage_pct'] < 30:
            observations.append("Low renewable energy usage contributes to higher carbon footprint.")
        
        if input_data['food_type'] == 'Non-Veg' and impact_level in ['Medium', 'High']:
            observations.append("Non-vegetarian diets typically have higher carbon footprints.")
        
        if input_data['eco_actions'] < 2:
            observations.append("Increasing eco-friendly actions can help reduce your footprint.")
        
        if observations:
            explanation_parts.append("\n**Key observations:**")
            for obs in observations:
                explanation_parts.append(f"- {obs}")
        
        # Join all parts
        return "\n".join(explanation_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        
        Returns:
            Dictionary with model information
        """
        info = {
            'total_models': len(self.models),
            'available_models': list(self.models.keys()),
            'preprocessor_loaded': self.preprocessor is not None,
            'label_encoder_loaded': self.label_encoder is not None,
            'metrics_loaded': bool(self.metrics)
        }
        
        # Add model types
        regression_models = [m for m in self.models.keys() if 'regression' in m.lower()]
        classification_models = [m for m in self.models.keys() if 'classification' in m.lower()]
        
        info['regression_models'] = regression_models
        info['classification_models'] = classification_models
        
        # Add performance metrics if available
        if self.metrics:
            if 'best_models' in self.metrics:
                info['best_models'] = self.metrics['best_models']
            
            if 'regression' in self.metrics:
                # Get best regression model performance
                best_reg = self.metrics.get('best_models', {}).get('regression', '')
                if best_reg and best_reg in self.metrics.get('regression', {}):
                    reg_metrics = self.metrics['regression'][best_reg]
                    info['regression_performance'] = {
                        'best_model': best_reg,
                        'r2': reg_metrics.get('r2', 0),
                        'rmse': reg_metrics.get('rmse', 0)
                    }
            
            if 'classification' in self.metrics:
                # Get best classification model performance
                best_clf = self.metrics.get('best_models', {}).get('classification', '')
                if best_clf and best_clf in self.metrics.get('classification', {}):
                    clf_metrics = self.metrics['classification'][best_clf]
                    info['classification_performance'] = {
                        'best_model': best_clf,
                        'accuracy': clf_metrics.get('accuracy', 0)
                    }
        
        return info
    
    def batch_predict(self, input_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple inputs
        
        Args:
            input_data_list: List of input dictionaries
        
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for i, input_data in enumerate(input_data_list):
            try:
                # Validate input
                is_valid, errors = self.validate_input(input_data)
                if not is_valid:
                    predictions.append({
                        'error': f"Validation failed: {', '.join(errors)}",
                        'index': i,
                        'carbon_footprint_kg': None,
                        'carbon_impact_level': 'Error'
                    })
                    continue
                
                # Make prediction
                pred = self.predict(input_data)
                pred['index'] = i
                predictions.append(pred)
                
            except Exception as e:
                predictions.append({
                    'error': str(e),
                    'index': i,
                    'carbon_footprint_kg': None,
                    'carbon_impact_level': 'Error'
                })
        
        return predictions
    
    def analyze_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a list of predictions
        
        Args:
            predictions: List of prediction dictionaries
        
        Returns:
            Analysis results
        """
        # Filter out errors
        valid_predictions = [p for p in predictions if 'error' not in p]
        error_predictions = [p for p in predictions if 'error' in p]
        
        analysis = {
            'total_predictions': len(predictions),
            'successful_predictions': len(valid_predictions),
            'failed_predictions': len(error_predictions),
            'success_rate': len(valid_predictions) / len(predictions) * 100 if predictions else 0
        }
        
        if valid_predictions:
            # Extract footprints
            footprints = [p['carbon_footprint_kg'] for p in valid_predictions]
            
            # Footprint statistics
            analysis['footprint_statistics'] = {
                'mean': float(np.mean(footprints)),
                'median': float(np.median(footprints)),
                'std': float(np.std(footprints)),
                'min': float(np.min(footprints)),
                'max': float(np.max(footprints)),
                'q1': float(np.percentile(footprints, 25)),
                'q3': float(np.percentile(footprints, 75))
            }
            
            # Impact level distribution
            impact_levels = [p['carbon_impact_level'] for p in valid_predictions]
            level_counts = {}
            for level in impact_levels:
                level_counts[level] = level_counts.get(level, 0) + 1
            
            analysis['impact_level_distribution'] = level_counts
            
            # Calculate percentages
            total = len(valid_predictions)
            analysis['impact_level_percentages'] = {
                level: count / total * 100
                for level, count in level_counts.items()
            }
            
            # Confidence statistics
            confidences = [p.get('confidence', 0) for p in valid_predictions]
            analysis['confidence_statistics'] = {
                'mean': float(np.mean(confidences)),
                'median': float(np.median(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            }
        
        if error_predictions:
            # Error analysis
            error_types = {}
            for pred in error_predictions:
                error_msg = pred['error']
                # Extract error type (simplified)
                if 'validation' in error_msg.lower():
                    error_type = 'Validation Error'
                elif 'model' in error_msg.lower():
                    error_type = 'Model Error'
                else:
                    error_type = 'Other Error'
                
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            analysis['error_analysis'] = error_types
        
        return analysis

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_predictor(models_dir: str = "models") -> ModelPredictor:
    """
    Load predictor from models directory
    
    Args:
        models_dir: Directory containing models
    
    Returns:
        ModelPredictor instance
    """
    return ModelPredictor(models_dir)

def make_quick_prediction(input_data: Dict[str, Any], 
                         models_dir: str = "models") -> Dict[str, Any]:
    """
    Make a quick prediction (convenience function)
    
    Args:
        input_data: Input data dictionary
        models_dir: Directory containing models
    
    Returns:
        Prediction dictionary
    """
    predictor = ModelPredictor(models_dir)
    return predictor.predict(input_data)

def validate_and_predict(input_data: Dict[str, Any], 
                        predictor: ModelPredictor) -> Dict[str, Any]:
    """
    Validate input and make prediction
    
    Args:
        input_data: Input data dictionary
        predictor: ModelPredictor instance
    
    Returns:
        Prediction dictionary or error dictionary
    """
    # Validate input
    is_valid, errors = predictor.validate_input(input_data)
    
    if not is_valid:
        return {
            'error': True,
            'error_messages': errors,
            'carbon_footprint_kg': None,
            'carbon_impact_level': 'Error'
        }
    
    # Make prediction
    try:
        return predictor.predict(input_data)
    except Exception as e:
        return {
            'error': True,
            'error_message': str(e),
            'carbon_footprint_kg': None,
            'carbon_impact_level': 'Error'
        }

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("Testing Carbon Footprint Predictor")
    print("=" * 50)
    
    # Sample input
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
    
    try:
        # Initialize predictor
        predictor = load_predictor()
        
        # Get model info
        model_info = predictor.get_model_info()
        print(f"\nModel Information:")
        print(f"  Total models: {model_info['total_models']}")
        print(f"  Regression models: {len(model_info['regression_models'])}")
        print(f"  Classification models: {len(model_info['classification_models'])}")
        
        # Make prediction
        print(f"\nMaking prediction for sample input...")
        prediction = predictor.predict(sample_input)
        
        print(f"\nPrediction Results:")
        print(f"  Carbon Footprint: {prediction['carbon_footprint_kg']:.2f} kg CO₂/day")
        print(f"  Impact Level: {prediction['carbon_impact_level']}")
        print(f"  Confidence: {prediction['confidence']:.2%}")
        
        print(f"\nProbabilities:")
        for level, prob in prediction.get('probabilities', {}).items():
            print(f"  {level}: {prob:.2%}")
        
        print(f"\nTop Suggestions:")
        for suggestion in prediction.get('suggestions', []):
            print(f"  [{suggestion['category']}] {suggestion['action']}")
            print(f"    Impact: {suggestion['impact']}")
            print(f"    Estimated reduction: {suggestion['estimated_reduction']}")
        
        print(f"\nFeature Contributions:")
        for feature, contribution in prediction.get('feature_contributions', {}).items():
            print(f"  {feature}: {contribution:.3f}")
        
        # Get explanation
        print(f"\nGetting detailed explanation...")
        explanation = predictor.explain_prediction(sample_input)
        
        print(f"\nExplanation:")
        print(explanation['explanation'])
        
        print(f"\nTop Factors:")
        for factor in explanation.get('top_factors', []):
            print(f"  {factor['factor']}: {factor['percentage']:.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()