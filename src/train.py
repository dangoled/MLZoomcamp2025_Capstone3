#!/usr/bin/env python3
"""
Training script for Carbon Footprint Prediction models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import pickle
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load data from CSV file"""
    df = pd.read_csv(filepath)
    return df

def prepare_features(df):
    """Prepare features and targets"""
    X = df.drop(['user_id', 'carbon_footprint_kg', 'carbon_impact_level'], axis=1)
    y_reg = df['carbon_footprint_kg']
    y_clf = df['carbon_impact_level']
    
    return X, y_reg, y_clf

def create_preprocessor():
    """Create preprocessing pipeline"""
    numeric_features = ['distance_km', 'electricity_kwh', 'renewable_usage_pct', 
                       'screen_time_hours', 'waste_generated_kg', 'eco_actions']
    
    categorical_features = ['day_type', 'transport_mode', 'food_type']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def train_regression_models(X_train, y_train, preprocessor):
    """Train multiple regression models"""
    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'xgboost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'lightgbm': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        'catboost': CatBoostRegressor(iterations=100, random_seed=42, verbose=0)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name} regression model...")
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
    
    return trained_models

def train_classification_models(X_train, y_train, preprocessor):
    """Train multiple classification models"""
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgboost': XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
        'lightgbm': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'catboost': CatBoostClassifier(iterations=100, random_seed=42, verbose=0)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name} classification model...")
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
    
    return trained_models

def evaluate_regression_models(models, X_test, y_test):
    """Evaluate regression models"""
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2)
        }
        
        print(f"\n{name.upper()} Regression:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
    
    return results

def evaluate_classification_models(models, X_test, y_test):
    """Evaluate classification models"""
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            'accuracy': float(accuracy),
            'classification_report': report
        }
        
        print(f"\n{name.upper()} Classification:")
        print(f"  Accuracy: {accuracy:.4f}")
    
    return results

def select_best_model(results, metric='r2', higher_is_better=True):
    """Select best model based on metric"""
    if higher_is_better:
        best_model = max(results.items(), key=lambda x: x[1][metric])
    else:
        best_model = min(results.items(), key=lambda x: x[1][metric])
    
    return best_model[0], best_model[1]

def save_models(models, preprocessor, metrics, output_dir):
    """Save models and metadata"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save preprocessor
    preprocessor_path = output_dir / 'preprocessor.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Save models
    for model_type, model_dict in models.items():
        for model_name, model in model_dict.items():
            model_path = output_dir / f'{model_type}_{model_name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nModels saved to {output_dir}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Train Carbon Footprint Prediction models')
    parser.add_argument('--data', type=str, default='data/raw/personal_carbon_footprint_behavior.csv',
                       help='Path to input data')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save models and metrics')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CARBON FOOTPRINT PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data(args.data)
    print(f"   Loaded {len(df)} records with {len(df.columns)} features")
    
    # Prepare features
    print("\n2. Preparing features...")
    X, y_reg, y_clf = prepare_features(df)
    
    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf, test_size=args.test_size, random_state=42, stratify=y_clf
    )
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Create preprocessor
    print("\n4. Creating preprocessing pipeline...")
    preprocessor = create_preprocessor()
    
    # Train regression models
    print("\n5. Training regression models...")
    regression_models = train_regression_models(X_train, y_train_reg, preprocessor)
    
    # Train classification models
    print("\n6. Training classification models...")
    classification_models = train_classification_models(X_train, y_train_clf, preprocessor)
    
    # Evaluate models
    print("\n7. Evaluating models...")
    print("\n" + "-" * 40)
    print("REGRESSION MODELS")
    print("-" * 40)
    regression_results = evaluate_regression_models(regression_models, X_test, y_test_reg)
    
    print("\n" + "-" * 40)
    print("CLASSIFICATION MODELS")
    print("-" * 40)
    classification_results = evaluate_classification_models(classification_models, X_test, y_test_clf)
    
    # Select best models
    best_reg_model, best_reg_metrics = select_best_model(regression_results, metric='r2')
    best_clf_model, best_clf_metrics = select_best_model(classification_results, metric='accuracy')
    
    print("\n" + "=" * 60)
    print("BEST MODELS SELECTED:")
    print(f"  Regression: {best_reg_model} (R²: {best_reg_metrics['r2']:.4f})")
    print(f"  Classification: {best_clf_model} (Accuracy: {best_clf_metrics['accuracy']:.4f})")
    print("=" * 60)
    
    # Prepare all models and metrics
    all_models = {
        'regression': regression_models,
        'classification': classification_models
    }
    
    all_metrics = {
        'regression': regression_results,
        'classification': classification_results,
        'best_models': {
            'regression': best_reg_model,
            'classification': best_clf_model
        }
    }
    
    # Save models
    print("\n8. Saving models and metrics...")
    output_path = save_models(all_models, preprocessor, all_metrics, args.output_dir)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Output saved to: {output_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()