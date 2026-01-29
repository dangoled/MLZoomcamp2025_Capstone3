"""
Model definitions and utilities for Carbon Footprint Prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import pickle
import json
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

# ============================================================================
# CUSTOM TRANSFORMERS
# ============================================================================

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer
    Creates new features based on domain knowledge
    """
    
    def __init__(self, create_interactions: bool = True):
        self.create_interactions = create_interactions
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create new engineered features"""
        X_transformed = X.copy()
        
        # 1. Create composite scores
        # Transport impact score (distance weighted by transport mode impact)
        transport_impact_map = {'Walk': 0.1, 'Bike': 0.2, 'Bus': 0.5, 'EV': 0.7, 'Car': 1.0}
        X_transformed['transport_impact_score'] = X_transformed['distance_km'] * \
            X_transformed['transport_mode'].map(transport_impact_map)
        
        # 2. Energy efficiency score
        # Lower is better (high renewable, low electricity)
        X_transformed['energy_efficiency'] = (
            X_transformed['electricity_kwh'] * 
            (100 - X_transformed['renewable_usage_pct']) / 100
        )
        
        # 3. Waste per activity
        X_transformed['waste_per_activity'] = X_transformed['waste_generated_kg'] / \
            (X_transformed['eco_actions'] + 1)  # +1 to avoid division by zero
        
        # 4. Sustainability score (higher is better)
        sustainability_components = []
        
        # Renewable energy component (0-1)
        sustainability_components.append(X_transformed['renewable_usage_pct'] / 100)
        
        # Eco actions component (0-1, normalized to max 5)
        sustainability_components.append(X_transformed['eco_actions'] / 5)
        
        # Transport component (inverse of impact)
        sustainability_components.append(1 - (X_transformed['transport_impact_score'] / 
                                            X_transformed['transport_impact_score'].max()))
        
        # Food type component
        food_scores = {'Veg': 1.0, 'Mixed': 0.5, 'Non-Veg': 0.0}
        sustainability_components.append(X_transformed['food_type'].map(food_scores))
        
        # Average sustainability score
        X_transformed['sustainability_score'] = np.mean(sustainability_components, axis=0)
        
        # 5. Create interaction features if requested
        if self.create_interactions:
            # Transport distance × electricity interaction
            X_transformed['transport_electricity_interaction'] = (
                X_transformed['distance_km'] * X_transformed['electricity_kwh']
            )
            
            # Renewable × waste interaction
            X_transformed['renewable_waste_interaction'] = (
                (100 - X_transformed['renewable_usage_pct']) * X_transformed['waste_generated_kg']
            )
        
        # 6. Day type encoding
        X_transformed['is_weekend'] = (X_transformed['day_type'] == 'Weekend').astype(int)
        
        return X_transformed

class CarbonFootprintThresholdTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to convert continuous carbon footprint to impact levels
    based on dataset statistics
    """
    
    def __init__(self, low_threshold: float = 6.5, high_threshold: float = 10.5):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.classes_ = np.array(['Low', 'Medium', 'High'])
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert carbon footprint to impact levels"""
        X_transformed = X.copy()
        
        if 'carbon_footprint_kg' in X.columns:
            # Create impact level based on thresholds
            conditions = [
                X_transformed['carbon_footprint_kg'] < self.low_threshold,
                X_transformed['carbon_footprint_kg'] <= self.high_threshold,
                X_transformed['carbon_footprint_kg'] > self.high_threshold
            ]
            choices = ['Low', 'Medium', 'High']
            X_transformed['carbon_impact_level'] = np.select(conditions, choices, default='Medium')
        
        return X_transformed

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for model training"""
    model_type: str  # 'regression' or 'classification'
    algorithm: str   # e.g., 'random_forest', 'xgboost', etc.
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_engineering: bool = True
    scale_features: bool = True
    use_cross_validation: bool = True
    cv_folds: int = 5
    random_state: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'model_type': self.model_type,
            'algorithm': self.algorithm,
            'hyperparameters': self.hyperparameters,
            'feature_engineering': self.feature_engineering,
            'scale_features': self.scale_features,
            'use_cross_validation': self.use_cross_validation,
            'cv_folds': self.cv_folds,
            'random_state': self.random_state
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary"""
        return cls(**config_dict)

class CarbonFootprintModel:
    """
    Main model class for Carbon Footprint Prediction
    Handles both regression and classification tasks
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.pipeline = None
        self.is_fitted = False
        self.feature_importance = None
        self.training_metrics = {}
        self.feature_names = None
        
    def build_pipeline(self, preprocessor) -> Pipeline:
        """
        Build the complete model pipeline
        Includes preprocessing, feature engineering, and modeling
        """
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from xgboost import XGBRegressor, XGBClassifier
        from lightgbm import LGBMRegressor, LGBMClassifier
        from catboost import CatBoostRegressor, CatBoostClassifier
        from sklearn.svm import SVR, SVC
        
        # Feature engineering step
        if self.config.feature_engineering:
            feature_engineering = FeatureEngineeringTransformer()
        else:
            feature_engineering = FunctionTransformer(lambda x: x)
        
        # Select model based on algorithm
        algorithm = self.config.algorithm.lower()
        
        if self.config.model_type == 'regression':
            if algorithm == 'random_forest':
                model = RandomForestRegressor(
                    random_state=self.config.random_state,
                    **self.config.hyperparameters
                )
            elif algorithm == 'linear_regression':
                model = LinearRegression(**self.config.hyperparameters)
            elif algorithm == 'xgboost':
                model = XGBRegressor(
                    random_state=self.config.random_state,
                    verbosity=0,
                    **self.config.hyperparameters
                )
            elif algorithm == 'lightgbm':
                model = LGBMRegressor(
                    random_state=self.config.random_state,
                    verbose=-1,
                    **self.config.hyperparameters
                )
            elif algorithm == 'catboost':
                model = CatBoostRegressor(
                    random_seed=self.config.random_state,
                    verbose=0,
                    **self.config.hyperparameters
                )
            elif algorithm == 'svr':
                model = SVR(**self.config.hyperparameters)
            else:
                raise ValueError(f"Unknown regression algorithm: {algorithm}")
        
        elif self.config.model_type == 'classification':
            if algorithm == 'random_forest':
                model = RandomForestClassifier(
                    random_state=self.config.random_state,
                    **self.config.hyperparameters
                )
            elif algorithm == 'logistic_regression':
                model = LogisticRegression(
                    random_state=self.config.random_state,
                    max_iter=1000,
                    **self.config.hyperparameters
                )
            elif algorithm == 'xgboost':
                model = XGBClassifier(
                    random_state=self.config.random_state,
                    verbosity=0,
                    **self.config.hyperparameters
                )
            elif algorithm == 'lightgbm':
                model = LGBMClassifier(
                    random_state=self.config.random_state,
                    verbose=-1,
                    **self.config.hyperparameters
                )
            elif algorithm == 'catboost':
                model = CatBoostClassifier(
                    random_seed=self.config.random_state,
                    verbose=0,
                    **self.config.hyperparameters
                )
            elif algorithm == 'svc':
                model = SVC(
                    random_state=self.config.random_state,
                    probability=True,
                    **self.config.hyperparameters
                )
            else:
                raise ValueError(f"Unknown classification algorithm: {algorithm}")
        
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        # Build the pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_engineering', feature_engineering),
            ('model', model)
        ])
        
        self.model = model
        return self.pipeline
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              preprocessor = None) -> Dict[str, Any]:
        """
        Train the model
        """
        from sklearn.model_selection import cross_val_score
        
        if preprocessor is None:
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline as SKPipeline
            
            # Default preprocessor
            numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
            
            numeric_transformer = SKPipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = SKPipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
        
        # Build and train pipeline
        self.build_pipeline(preprocessor)
        
        if self.config.use_cross_validation:
            # Perform cross-validation
            if self.config.model_type == 'regression':
                scoring = 'r2'
            else:  # classification
                scoring = 'accuracy'
            
            cv_scores = cross_val_score(
                self.pipeline, X_train, y_train,
                cv=self.config.cv_folds,
                scoring=scoring,
                n_jobs=-1
            )
            
            self.training_metrics['cv_mean'] = cv_scores.mean()
            self.training_metrics['cv_std'] = cv_scores.std()
            self.training_metrics['cv_scores'] = cv_scores.tolist()
        
        # Fit the model on full training data
        self.pipeline.fit(X_train, y_train)
        
        # Extract feature importance if available
        self._extract_feature_importance(X_train.columns)
        
        # Get feature names after preprocessing
        self._extract_feature_names()
        
        self.is_fitted = True
        
        # Calculate training metrics
        if self.config.model_type == 'regression':
            from sklearn.metrics import mean_squared_error, r2_score
            y_pred_train = self.predict(X_train)
            self.training_metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_pred_train))
            self.training_metrics['train_r2'] = r2_score(y_train, y_pred_train)
        else:
            from sklearn.metrics import accuracy_score
            y_pred_train = self.predict(X_train)
            self.training_metrics['train_accuracy'] = accuracy_score(y_train, y_pred_train)
        
        # Calculate validation metrics if validation data provided
        if X_val is not None and y_val is not None:
            if self.config.model_type == 'regression':
                from sklearn.metrics import mean_squared_error, r2_score
                y_pred_val = self.predict(X_val)
                self.training_metrics['val_rmse'] = np.sqrt(mean_squared_error(y_val, y_pred_val))
                self.training_metrics['val_r2'] = r2_score(y_val, y_pred_val)
            else:
                from sklearn.metrics import accuracy_score, classification_report
                y_pred_val = self.predict(X_val)
                self.training_metrics['val_accuracy'] = accuracy_score(y_val, y_pred_val)
                self.training_metrics['val_report'] = classification_report(
                    y_val, y_pred_val, output_dict=True
                )
        
        return self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (for classification models only)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if self.config.model_type != 'classification':
            raise ValueError("predict_proba is only available for classification models")
        
        if hasattr(self.model, 'predict_proba'):
            return self.pipeline.predict_proba(X)
        else:
            raise AttributeError(f"Model {self.config.algorithm} does not support predict_proba")
    
    def _extract_feature_importance(self, original_features: List[str]):
        """
        Extract feature importance from the model if available
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # Get feature names after preprocessing and engineering
            if hasattr(self.pipeline.named_steps['feature_engineering'], 'get_feature_names_out'):
                feature_names = self.pipeline.named_steps['feature_engineering'].get_feature_names_out()
            else:
                # Fallback: use indices
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
    
    def _extract_feature_names(self):
        """
        Extract feature names after all transformations
        """
        # This is a simplified version - in practice would need to traverse pipeline
        try:
            # For preprocessor
            if hasattr(self.pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
                preprocessed_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out()
            else:
                preprocessed_names = None
            
            # For feature engineering
            if hasattr(self.pipeline.named_steps['feature_engineering'], 'get_feature_names_out'):
                final_names = self.pipeline.named_steps['feature_engineering'].get_feature_names_out()
            else:
                final_names = preprocessed_names
            
            self.feature_names = final_names
        except:
            self.feature_names = None
    
    def save(self, filepath: Union[str, Path]):
        """
        Save the model to disk
        """
        model_data = {
            'config': self.config.to_dict(),
            'pipeline': self.pipeline,
            'is_fitted': self.is_fitted,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'CarbonFootprintModel':
        """
        Load a model from disk
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        config = ModelConfig.from_dict(model_data['config'])
        model = cls(config)
        
        model.pipeline = model_data['pipeline']
        model.is_fitted = model_data['is_fitted']
        model.feature_importance = model_data['feature_importance']
        model.training_metrics = model_data['training_metrics']
        model.feature_names = model_data['feature_names']
        
        # Get model from pipeline
        if model.pipeline is not None:
            model.model = model.pipeline.named_steps.get('model', None)
        
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model
        """
        info = {
            'model_type': self.config.model_type,
            'algorithm': self.config.algorithm,
            'is_fitted': self.is_fitted,
            'training_metrics': self.training_metrics,
            'feature_engineering': self.config.feature_engineering,
            'hyperparameters': self.config.hyperparameters
        }
        
        if self.feature_importance is not None:
            info['top_features'] = self.feature_importance.head(10).to_dict('records')
        
        return info

# ============================================================================
# MODEL FACTORY
# ============================================================================

class ModelFactory:
    """
    Factory class for creating Carbon Footprint models
    """
    
    @staticmethod
    def get_default_configs() -> Dict[str, ModelConfig]:
        """
        Get default model configurations
        """
        return {
            # Regression models
            'random_forest_reg': ModelConfig(
                model_type='regression',
                algorithm='random_forest',
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'random_state': 42
                }
            ),
            'xgboost_reg': ModelConfig(
                model_type='regression',
                algorithm='xgboost',
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            ),
            'lightgbm_reg': ModelConfig(
                model_type='regression',
                algorithm='lightgbm',
                hyperparameters={
                    'n_estimators': 100,
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            ),
            
            # Classification models
            'random_forest_clf': ModelConfig(
                model_type='classification',
                algorithm='random_forest',
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'random_state': 42
                }
            ),
            'xgboost_clf': ModelConfig(
                model_type='classification',
                algorithm='xgboost',
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            ),
            'lightgbm_clf': ModelConfig(
                model_type='classification',
                algorithm='lightgbm',
                hyperparameters={
                    'n_estimators': 100,
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            )
        }
    
    @staticmethod
    def create_model(model_name: str, custom_config: Optional[Dict[str, Any]] = None) -> CarbonFootprintModel:
        """
        Create a model by name
        """
        default_configs = ModelFactory.get_default_configs()
        
        if model_name not in default_configs:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(default_configs.keys())}")
        
        config = default_configs[model_name]
        
        if custom_config:
            # Update config with custom values
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                elif key == 'hyperparameters':
                    config.hyperparameters.update(value)
        
        return CarbonFootprintModel(config)
    
    @staticmethod
    def create_ensemble(models: List[CarbonFootprintModel], ensemble_type: str = 'voting') -> 'ModelEnsemble':
        """
        Create an ensemble of models
        """
        return ModelEnsemble(models, ensemble_type)

# ============================================================================
# MODEL ENSEMBLE
# ============================================================================

class ModelEnsemble:
    """
    Ensemble of multiple models
    """
    
    def __init__(self, models: List[CarbonFootprintModel], ensemble_type: str = 'voting'):
        self.models = models
        self.ensemble_type = ensemble_type
        self.is_fitted = all(model.is_fitted for model in models)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("All models in ensemble must be trained")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if self.ensemble_type == 'voting':
            # For classification: majority vote
            if self.models[0].config.model_type == 'classification':
                from scipy import stats
                return stats.mode(predictions, axis=0)[0].flatten()
            # For regression: average
            else:
                return predictions.mean(axis=0)
        
        elif self.ensemble_type == 'weighted':
            # Weighted average (weights could be based on model performance)
            weights = np.array([model.training_metrics.get('val_r2', 0.5) 
                              if model.config.model_type == 'regression'
                              else model.training_metrics.get('val_accuracy', 0.5)
                              for model in self.models])
            weights = weights / weights.sum()
            return np.average(predictions, axis=0, weights=weights)
        
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (for classification ensembles)
        """
        if not self.is_fitted:
            raise ValueError("All models in ensemble must be trained")
        
        if self.models[0].config.model_type != 'classification':
            raise ValueError("predict_proba is only available for classification models")
        
        probas = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probas.append(proba)
        
        if not probas:
            raise ValueError("None of the models support predict_proba")
        
        probas = np.array(probas)
        
        if self.ensemble_type == 'voting':
            return probas.mean(axis=0)
        elif self.ensemble_type == 'weighted':
            weights = np.array([model.training_metrics.get('val_accuracy', 0.5)
                              for model in self.models])
            weights = weights / weights.sum()
            # Weighted average of probabilities
            return np.average(probas, axis=0, weights=weights[:, np.newaxis, np.newaxis])
        else:
            return probas.mean(axis=0)

# ============================================================================
# MODEL EVALUATION UTILITIES
# ============================================================================

class ModelEvaluator:
    """
    Utility class for evaluating model performance
    """
    
    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression model performance
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate additional metrics
        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Calculate percentage within error bounds
        within_1kg = np.mean(np.abs(residuals) <= 1.0) * 100
        within_2kg = np.mean(np.abs(residuals) <= 2.0) * 100
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mean_residual': float(mean_residual),
            'std_residual': float(std_residual),
            'within_1kg_pct': float(within_1kg),
            'within_2kg_pct': float(within_2kg),
            'explained_variance': float(r2)
        }
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, 
                               y_proba: Optional[np.ndarray] = None,
                               class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate classification model performance
        """
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                   f1_score, confusion_matrix, classification_report,
                                   roc_auc_score)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        results = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
        }
        
        # Per-class metrics
        if class_names is not None:
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            results['per_class'] = {}
            for i, class_name in enumerate(class_names):
                results['per_class'][class_name] = {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1': float(f1_per_class[i])
                }
        
        # ROC AUC if probabilities are available
        if y_proba is not None:
            try:
                # For multi-class, need to handle one-vs-rest or one-vs-one
                if len(np.unique(y_true)) > 2:
                    roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                else:
                    roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                results['roc_auc'] = float(roc_auc)
            except:
                results['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        results['classification_report'] = report
        
        return results
    
    @staticmethod
    def compare_models(models: List[CarbonFootprintModel], 
                      X_test: pd.DataFrame, 
                      y_test: pd.Series) -> pd.DataFrame:
        """
        Compare multiple models on test data
        """
        comparisons = []
        
        for model in models:
            if not model.is_fitted:
                continue
            
            y_pred = model.predict(X_test)
            
            if model.config.model_type == 'regression':
                metrics = ModelEvaluator.evaluate_regression(y_test.values, y_pred)
                metrics['model_type'] = 'regression'
            else:
                # Try to get probabilities for classification
                y_proba = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X_test)
                    except:
                        pass
                
                metrics = ModelEvaluator.evaluate_classification(
                    y_test.values, y_pred, y_proba
                )
                metrics['model_type'] = 'classification'
            
            metrics['model_name'] = f"{model.config.algorithm}_{model.config.model_type}"
            metrics['algorithm'] = model.config.algorithm
            
            comparisons.append(metrics)
        
        return pd.DataFrame(comparisons)

# ============================================================================
# MODEL EXPLAINABILITY
# ============================================================================

class ModelExplainer:
    """
    Class for explaining model predictions
    """
    
    def __init__(self, model: CarbonFootprintModel):
        self.model = model
    
    def explain_prediction(self, X: pd.DataFrame, sample_index: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction
        """
        if not self.model.is_fitted:
            raise ValueError("Model must be trained before explaining predictions")
        
        if isinstance(X, pd.DataFrame):
            X_sample = X.iloc[[sample_index]]
        else:
            X_sample = X[[sample_index]]
        
        # Get prediction
        prediction = self.model.predict(X_sample)[0]
        
        # Get feature importance for this sample if available
        feature_contributions = self._get_feature_contributions(X_sample)
        
        # Get SHAP values if available
        shap_values = self._get_shap_values(X_sample)
        
        explanation = {
            'prediction': float(prediction) if self.model.config.model_type == 'regression' else int(prediction),
            'feature_contributions': feature_contributions,
            'shap_values': shap_values,
            'feature_values': X_sample.iloc[0].to_dict() if isinstance(X_sample, pd.DataFrame) else {}
        }
        
        if self.model.config.model_type == 'classification':
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_sample)[0]
                explanation['probabilities'] = {str(i): float(p) for i, p in enumerate(proba)}
        
        return explanation
    
    def _get_feature_contributions(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Get feature contributions to prediction
        Simplified version - in production would use SHAP or LIME
        """
        contributions = {}
        
        if self.model.feature_importance is not None and self.model.feature_names is not None:
            # Use feature importance as proxy for contributions
            for idx, row in self.model.feature_importance.head(10).iterrows():
                feature_name = row['feature']
                importance = row['importance']
                
                # Try to get feature value
                try:
                    # This is simplified - actual implementation would map engineered features
                    contributions[feature_name] = float(importance)
                except:
                    contributions[feature_name] = float(importance)
        
        return contributions
    
    def _get_shap_values(self, X: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Calculate SHAP values if shap library is available
        """
        try:
            import shap
            
            # Check if model supports TreeExplainer
            model_type = type(self.model.model).__name__.lower()
            
            if 'randomforest' in model_type or 'xgboost' in model_type or 'lightgbm' in model_type:
                # Use TreeExplainer for tree-based models
                explainer = shap.TreeExplainer(self.model.model)
                shap_values = explainer.shap_values(X)
                
                if isinstance(shap_values, list):
                    # Multi-class classification
                    return {f'class_{i}': vals.tolist() for i, vals in enumerate(shap_values)}
                else:
                    # Regression or binary classification
                    return {'values': shap_values.tolist()}
            
            else:
                # Use KernelExplainer for other models
                background = shap.kmeans(X, 10)
                explainer = shap.KernelExplainer(self.model.model.predict, background)
                shap_values = explainer.shap_values(X)
                
                return {'values': shap_values.tolist()}
                
        except ImportError:
            return None
        except Exception as e:
            print(f"SHAP calculation failed: {e}")
            return None
    
    def get_global_explanation(self) -> Dict[str, Any]:
        """
        Get global model explanation (feature importance)
        """
        if not self.model.is_fitted:
            raise ValueError("Model must be trained before explaining")
        
        explanation = {
            'model_type': self.model.config.model_type,
            'algorithm': self.model.config.algorithm,
            'feature_engineering': self.model.config.feature_engineering
        }
        
        if self.model.feature_importance is not None:
            explanation['feature_importance'] = self.model.feature_importance.head(20).to_dict('records')
        
        if self.model.training_metrics:
            explanation['performance_metrics'] = self.model.training_metrics
        
        return explanation

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_model_from_config(config_file: Union[str, Path]) -> CarbonFootprintModel:
    """
    Create model from configuration file
    """
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    config = ModelConfig.from_dict(config_dict)
    return CarbonFootprintModel(config)

def save_model_with_metadata(model: CarbonFootprintModel, filepath: Union[str, Path],
                            metadata: Optional[Dict[str, Any]] = None):
    """
    Save model with metadata
    """
    model.save(filepath)
    
    # Save metadata separately
    if metadata:
        metadata_file = Path(filepath).with_suffix('.metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

def load_model_with_metadata(filepath: Union[str, Path]) -> Tuple[CarbonFootprintModel, Dict[str, Any]]:
    """
    Load model with metadata
    """
    model = CarbonFootprintModel.load(filepath)
    
    # Load metadata if available
    metadata_file = Path(filepath).with_suffix('.metadata.json')
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata