"""
Tests for prediction utilities
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.predict import (
    ModelPredictor,
    load_predictor,
    make_quick_prediction,
    validate_and_predict
)

class TestModelPredictor:
    """Test ModelPredictor class"""
    
    def test_initialization(self, tmp_path):
        """Test predictor initialization"""
        with patch('src.predict.ModelPredictor._load_models'):
            predictor = ModelPredictor(str(tmp_path))
            
            assert predictor.models_dir == tmp_path
            assert predictor.models == {}
            assert predictor.preprocessor is None
            assert predictor.label_encoder is None
            assert predictor.metrics == {}
    
    def test_validate_input_valid(self, test_predictor, sample_input_data):
        """Test input validation with valid data"""
        is_valid, errors = test_predictor.validate_input(sample_input_data)
        
        assert is_valid is True
        assert errors == []
    
    @pytest.mark.parametrize("field,value,expected_error", [
        ('day_type', 'InvalidDay', "day_type must be one of: ['Weekday', 'Weekend']"),
        ('transport_mode', 'InvalidTransport', "transport_mode must be one of: ['Walk', 'Bike', 'Bus', 'EV', 'Car']"),
        ('food_type', 'InvalidFood', "food_type must be one of: ['Veg', 'Mixed', 'Non-Veg']"),
        ('distance_km', -5.0, "distance_km must be between 0 and 100"),
        ('distance_km', 150.0, "distance_km must be between 0 and 100"),
        ('electricity_kwh', -1.0, "electricity_kwh must be between 0 and 50"),
        ('renewable_usage_pct', -10.0, "renewable_usage_pct must be between 0 and 100"),
        ('renewable_usage_pct', 150.0, "renewable_usage_pct must be between 0 and 100"),
        ('screen_time_hours', 25.0, "screen_time_hours must be between 0 and 24"),
        ('waste_generated_kg', -0.5, "waste_generated_kg must be between 0 and 10"),
        ('eco_actions', -1, "eco_actions must be between 0 and 10"),
        ('eco_actions', 15, "eco_actions must be between 0 and 10"),
    ])
    def test_validate_input_invalid(self, test_predictor, sample_input_data, 
                                   field, value, expected_error):
        """Test input validation with invalid data"""
        invalid_data = sample_input_data.copy()
        invalid_data[field] = value
        
        is_valid, errors = test_predictor.validate_input(invalid_data)
        
        assert is_valid is False
        assert expected_error in errors[0]
    
    def test_validate_input_missing_field(self, test_predictor, sample_input_data):
        """Test input validation with missing field"""
        invalid_data = sample_input_data.copy()
        del invalid_data['day_type']  # Remove required field
        
        is_valid, errors = test_predictor.validate_input(invalid_data)
        
        assert is_valid is False
        assert "Missing required field: day_type" in errors
    
    def test_prepare_input_dataframe(self, test_predictor, sample_input_data):
        """Test preparing input DataFrame"""
        df = test_predictor.prepare_input_dataframe(sample_input_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        
        expected_columns = [
            'day_type', 'transport_mode', 'distance_km', 
            'electricity_kwh', 'renewable_usage_pct', 'food_type',
            'screen_time_hours', 'waste_generated_kg', 'eco_actions'
        ]
        
        assert list(df.columns) == expected_columns
        
        # Check values
        for key, value in sample_input_data.items():
            assert df[key].iloc[0] == value
    
    def test_predict_regression(self, test_predictor, sample_input_data):
        """Test regression prediction"""
        input_df = test_predictor.prepare_input_dataframe(sample_input_data)
        
        prediction, confidence = test_predictor.predict_regression(input_df)
        
        assert isinstance(prediction, float)
        assert 0 <= confidence <= 1
        assert 1 <= prediction <= 20  # Reasonable carbon footprint range
    
    def test_predict_regression_model_not_found(self, test_predictor, sample_input_data):
        """Test regression prediction with non-existent model"""
        input_df = test_predictor.prepare_input_dataframe(sample_input_data)
        
        with pytest.raises(ValueError, match="Model nonexistent not found"):
            test_predictor.predict_regression(input_df, 'nonexistent')
    
    def test_predict_classification(self, test_predictor, sample_input_data):
        """Test classification prediction"""
        input_df = test_predictor.prepare_input_dataframe(sample_input_data)
        
        impact_level, confidence, probabilities = test_predictor.predict_classification(input_df)
        
        assert impact_level in ['Low', 'Medium', 'High']
        assert 0 <= confidence <= 1
        assert isinstance(probabilities, dict)
        assert set(probabilities.keys()) == {'Low', 'Medium', 'High'}
        
        # Probabilities should sum to approximately 1
        prob_sum = sum(probabilities.values())
        assert abs(prob_sum - 1.0) < 0.01
    
    def test_predict_complete(self, test_predictor, sample_input_data):
        """Test complete prediction pipeline"""
        result = test_predictor.predict(sample_input_data)
        
        # Check required fields
        assert 'carbon_footprint_kg' in result
        assert 'carbon_impact_level' in result
        assert 'confidence' in result
        assert 'suggestions' in result
        assert 'feature_contributions' in result
        assert 'timestamp' in result
        assert 'input_summary' in result
        
        # Check types
        assert isinstance(result['carbon_footprint_kg'], float)
        assert result['carbon_impact_level'] in ['Low', 'Medium', 'High']
        assert isinstance(result['confidence'], float)
        assert isinstance(result['suggestions'], list)
        assert isinstance(result['feature_contributions'], dict)
        assert isinstance(result['timestamp'], str)
        assert isinstance(result['input_summary'], dict)
        
        # Check value ranges
        assert 1 <= result['carbon_footprint_kg'] <= 20
        assert 0 <= result['confidence'] <= 1
    
    def test_predict_with_invalid_input(self, test_predictor):
        """Test prediction with invalid input"""
        invalid_data = {'day_type': 'InvalidDay'}  # Missing other fields
        
        with pytest.raises(ValueError, match="Invalid input"):
            test_predictor.predict(invalid_data)
    
    def test_explain_prediction(self, test_predictor, sample_input_data):
        """Test prediction explanation"""
        explanation = test_predictor.explain_prediction(sample_input_data)
        
        # Check structure
        assert 'prediction' in explanation
        assert 'feature_contributions' in explanation
        assert 'percentage_contributions' in explanation
        assert 'top_factors' in explanation
        assert 'explanation' in explanation
        assert 'suggestions' in explanation
        
        # Check prediction sub-dictionary
        pred = explanation['prediction']
        assert 'carbon_footprint_kg' in pred
        assert 'carbon_impact_level' in pred
        assert 'confidence' in pred
        
        # Check top factors
        assert isinstance(explanation['top_factors'], list)
        if explanation['top_factors']:
            factor = explanation['top_factors'][0]
            assert 'factor' in factor
            assert 'percentage' in factor
        
        # Check explanation text
        assert isinstance(explanation['explanation'], str)
        assert len(explanation['explanation']) > 0
    
    def test_batch_predict(self, test_predictor, sample_input_data):
        """Test batch predictions"""
        batch_inputs = [sample_input_data, sample_input_data, sample_input_data]
        
        predictions = test_predictor.batch_predict(batch_inputs)
        
        assert len(predictions) == 3
        
        for i, pred in enumerate(predictions):
            assert 'carbon_footprint_kg' in pred
            assert 'carbon_impact_level' in pred
            assert pred['index'] == i
    
    def test_batch_predict_with_errors(self, test_predictor):
        """Test batch predictions with some invalid inputs"""
        valid_input = {
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
        
        invalid_input = {'day_type': 'InvalidDay'}  # Missing other fields
        
        batch_inputs = [valid_input, invalid_input, valid_input]
        
        predictions = test_predictor.batch_predict(batch_inputs)
        
        assert len(predictions) == 3
        
        # First should be valid
        assert 'error' not in predictions[0]
        
        # Second should have error
        assert 'error' in predictions[1]
        
        # Third should be valid
        assert 'error' not in predictions[2]
    
    def test_analyze_predictions(self, test_predictor):
        """Test prediction analysis"""
        predictions = [
            {'carbon_footprint_kg': 5.0, 'carbon_impact_level': 'Low', 'confidence': 0.9},
            {'carbon_footprint_kg': 8.0, 'carbon_impact_level': 'Medium', 'confidence': 0.8},
            {'carbon_footprint_kg': 12.0, 'carbon_impact_level': 'High', 'confidence': 0.85},
            {'error': 'Validation failed'}
        ]
        
        analysis = test_predictor.analyze_predictions(predictions)
        
        assert 'total_predictions' in analysis
        assert 'successful_predictions' in analysis
        assert 'failed_predictions' in analysis
        assert 'success_rate' in analysis
        
        assert analysis['total_predictions'] == 4
        assert analysis['successful_predictions'] == 3
        assert analysis['failed_predictions'] == 1
        assert analysis['success_rate'] == 75.0
        
        # Check footprint statistics
        if 'footprint_statistics' in analysis:
            stats = analysis['footprint_statistics']
            assert 'mean' in stats
            assert 'median' in stats
            assert 'std' in stats
        
        # Check impact level distribution
        if 'impact_level_distribution' in analysis:
            dist = analysis['impact_level_distribution']
            assert 'Low' in dist
            assert 'Medium' in dist
            assert 'High' in dist
    
    def test_get_model_info(self, test_predictor):
        """Test getting model information"""
        info = test_predictor.get_model_info()
        
        assert 'total_models' in info
        assert 'available_models' in info
        assert 'regression_models' in info
        assert 'classification_models' in info
        assert 'preprocessor_loaded' in info
        assert 'label_encoder_loaded' in info
        assert 'metrics_loaded' in info
    
    def test_generate_suggestions(self, test_predictor, sample_input_data):
        """Test suggestion generation"""
        # Test High impact
        suggestions = test_predictor._generate_suggestions(
            sample_input_data, 'High', 12.0
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3
        
        if suggestions:
            suggestion = suggestions[0]
            assert 'category' in suggestion
            assert 'action' in suggestion
            assert 'impact' in suggestion
            assert 'estimated_reduction' in suggestion
            assert 'reason' in suggestion
        
        # Test Medium impact
        suggestions = test_predictor._generate_suggestions(
            sample_input_data, 'Medium', 8.0
        )
        
        # Test Low impact
        suggestions = test_predictor._generate_suggestions(
            sample_input_data, 'Low', 4.0
        )

class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @patch('src.predict.ModelPredictor')
    def test_load_predictor(self, mock_predictor_class):
        """Test load_predictor function"""
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor
        
        predictor = load_predictor('models')
        
        mock_predictor_class.assert_called_once_with('models')
        assert predictor == mock_predictor
    
    @patch('src.predict.ModelPredictor')
    def test_make_quick_prediction(self, mock_predictor_class):
        """Test make_quick_prediction function"""
        mock_predictor = Mock()
        mock_predictor.predict.return_value = {'carbon_footprint_kg': 8.25}
        mock_predictor_class.return_value = mock_predictor
        
        sample_input = {'test': 'data'}
        result = make_quick_prediction(sample_input, 'models')
        
        mock_predictor_class.assert_called_once_with('models')
        mock_predictor.predict.assert_called_once_with(sample_input)
        assert result == {'carbon_footprint_kg': 8.25}
    
    def test_validate_and_predict_valid(self, test_predictor, sample_input_data):
        """Test validate_and_predict with valid input"""
        result = validate_and_predict(sample_input_data, test_predictor)
        
        assert 'error' not in result
        assert 'carbon_footprint_kg' in result
    
    def test_validate_and_predict_invalid(self, test_predictor):
        """Test validate_and_predict with invalid input"""
        invalid_data = {'day_type': 'InvalidDay'}
        
        result = validate_and_predict(invalid_data, test_predictor)
        
        assert 'error' in result
        assert result['error'] is True
        assert 'error_messages' in result
    
    def test_validate_and_predict_exception(self, test_predictor, sample_input_data):
        """Test validate_and_predict when prediction fails"""
        # Mock prediction to raise exception
        test_predictor.predict = Mock(side_effect=Exception("Prediction failed"))
        
        result = validate_and_predict(sample_input_data, test_predictor)
        
        assert 'error' in result
        assert result['error'] is True
        assert 'error_message' in result

class TestEdgeCases:
    """Test edge cases"""
    
    def test_edge_case_values(self, test_predictor, edge_case_inputs):
        """Test predictions with edge case values"""
        for input_data in edge_case_inputs:
            result = test_predictor.predict(input_data)
            
            # Basic validation
            assert 'carbon_footprint_kg' in result
            assert 'carbon_impact_level' in result
            assert 'confidence' in result
    
    def test_mixed_valid_invalid_batch(self, test_predictor, sample_input_data):
        """Test batch prediction with mixed valid and invalid inputs"""
        batch = [
            sample_input_data,  # Valid
            {'invalid': 'data'},  # Invalid
            sample_input_data   # Valid
        ]
        
        results = test_predictor.batch_predict(batch)
        
        assert len(results) == 3
        assert 'error' not in results[0]  # First valid
        assert 'error' in results[1]      # Second invalid
        assert 'error' not in results[2]  # Third valid
    
    @pytest.mark.slow
    def test_performance_large_batch(self, test_predictor, sample_input_data):
        """Test performance with large batch (marked as slow)"""
        # Create large batch
        large_batch = [sample_input_data] * 1000
        
        import time
        start_time = time.time()
        
        results = test_predictor.batch_predict(large_batch)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete in reasonable time
        assert execution_time < 10.0  # 10 seconds max
        assert len(results) == 1000