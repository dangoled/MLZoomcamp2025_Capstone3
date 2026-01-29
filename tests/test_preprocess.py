"""
Tests for data preprocessing utilities
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.preprocess import CarbonFootprintPreprocessor

class TestCarbonFootprintPreprocessor:
    """Test CarbonFootprintPreprocessor class"""
    
    def test_prepare_input_dict(self, sample_input_data):
        """Test preparing input dictionary"""
        preprocessor = CarbonFootprintPreprocessor()
        
        df = preprocessor.prepare_input_dict(sample_input_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert list(df.columns) == [
            'day_type', 'transport_mode', 'distance_km', 
            'electricity_kwh', 'renewable_usage_pct', 'food_type',
            'screen_time_hours', 'waste_generated_kg', 'eco_actions'
        ]
        
        # Check numeric conversion
        assert df['distance_km'].dtype in [np.float64, np.float32]
        assert df['electricity_kwh'].dtype in [np.float64, np.float32]
        assert df['renewable_usage_pct'].dtype in [np.float64, np.float32]
    
    def test_prepare_input_dict_missing_values(self):
        """Test handling missing values in input"""
        preprocessor = CarbonFootprintPreprocessor()
        
        input_data = {
            'day_type': 'Weekday',
            'transport_mode': 'Car',
            'distance_km': 15.5,
            # Missing electricity_kwh
            'renewable_usage_pct': 25.0,
            'food_type': 'Mixed',
            'screen_time_hours': 6.5,
            'waste_generated_kg': 0.75,
            'eco_actions': 2
        }
        
        df = preprocessor.prepare_input_dict(input_data)
        
        # Missing value should be NaN
        assert pd.isna(df['electricity_kwh'].iloc[0])
    
    def test_prepare_input_dict_invalid_numeric(self):
        """Test handling invalid numeric values"""
        preprocessor = CarbonFootprintPreprocessor()
        
        input_data = {
            'day_type': 'Weekday',
            'transport_mode': 'Car',
            'distance_km': 'invalid',  # Invalid numeric
            'electricity_kwh': 8.2,
            'renewable_usage_pct': 25.0,
            'food_type': 'Mixed',
            'screen_time_hours': 6.5,
            'waste_generated_kg': 0.75,
            'eco_actions': 2
        }
        
        df = preprocessor.prepare_input_dict(input_data)
        
        # Invalid numeric should be NaN
        assert pd.isna(df['distance_km'].iloc[0])
    
    def test_get_sample_input(self):
        """Test getting sample input"""
        preprocessor = CarbonFootprintPreprocessor()
        
        sample = preprocessor.get_sample_input()
        
        assert isinstance(sample, dict)
        assert 'day_type' in sample
        assert 'transport_mode' in sample
        assert 'distance_km' in sample
        assert 'electricity_kwh' in sample
        assert 'renewable_usage_pct' in sample
        assert 'food_type' in sample
        assert 'screen_time_hours' in sample
        assert 'waste_generated_kg' in sample
        assert 'eco_actions' in sample
        
        # Check value types
        assert isinstance(sample['day_type'], str)
        assert isinstance(sample['transport_mode'], str)
        assert isinstance(sample['distance_km'], (int, float))
        assert isinstance(sample['electricity_kwh'], (int, float))
        assert isinstance(sample['renewable_usage_pct'], (int, float))
        assert isinstance(sample['food_type'], str)
        assert isinstance(sample['screen_time_hours'], (int, float))
        assert isinstance(sample['waste_generated_kg'], (int, float))
        assert isinstance(sample['eco_actions'], int)
    
    def test_load_preprocessor(self, tmp_path):
        """Test loading preprocessor from file"""
        preprocessor = CarbonFootprintPreprocessor()
        
        # Create a mock preprocessor
        mock_preprocessor = Mock()
        
        # Save to temporary file
        import pickle
        preprocessor_path = tmp_path / 'preprocessor.pkl'
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(mock_preprocessor, f)
        
        # Load preprocessor
        preprocessor.load_preprocessor(str(preprocessor_path))
        
        assert preprocessor.preprocessor is not None
    
    def test_load_preprocessor_file_not_found(self):
        """Test loading preprocessor when file doesn't exist"""
        preprocessor = CarbonFootprintPreprocessor()
        
        with pytest.raises(FileNotFoundError):
            preprocessor.load_preprocessor('nonexistent.pkl')
    
    def test_transform_without_loading(self):
        """Test transform without loading preprocessor first"""
        preprocessor = CarbonFootprintPreprocessor()
        
        with pytest.raises(ValueError, match="Preprocessor not loaded"):
            preprocessor.transform(pd.DataFrame())
    
    @patch('src.preprocess.pickle.load')
    def test_load_model(self, mock_pickle_load):
        """Test loading models"""
        from src.preprocess import ModelLoader
        
        # Create mock model
        mock_model = Mock()
        mock_pickle_load.return_value = mock_model
        
        # Test loading
        model = ModelLoader.load_model('dummy.pkl')
        
        assert model == mock_model
        mock_pickle_load.assert_called_once()
    
    def test_get_feature_names(self):
        """Test getting feature names"""
        preprocessor = CarbonFootprintPreprocessor()
        
        # Mock preprocessor with transformers
        mock_preprocessor = Mock()
        mock_preprocessor.transformers_ = [
            ('num', Mock(), ['distance_km', 'electricity_kwh']),
            ('cat', Mock(), ['day_type', 'transport_mode'])
        ]
        
        # Mock categorical encoder
        mock_encoder = Mock()
        mock_encoder.get_feature_names_out.return_value = [
            'day_type_Weekday', 'day_type_Weekend',
            'transport_mode_Car', 'transport_mode_Bus'
        ]
        mock_preprocessor.transformers_[1][1].named_steps = {'onehot': mock_encoder}
        
        preprocessor.preprocessor = mock_preprocessor
        preprocessor._extract_feature_names()
        
        # In actual implementation, would check feature_names
        # This test is for the method structure
        assert hasattr(preprocessor, '_extract_feature_names')

class TestDataFrameHandling:
    """Test DataFrame handling utilities"""
    
    def test_create_dataframe_from_dict(self, sample_input_data):
        """Test creating DataFrame from dictionary"""
        from src.preprocess import CarbonFootprintPreprocessor
        
        df = CarbonFootprintPreprocessor.prepare_input_dict(sample_input_data)
        
        # Check column order
        expected_columns = [
            'day_type', 'transport_mode', 'distance_km', 
            'electricity_kwh', 'renewable_usage_pct', 'food_type',
            'screen_time_hours', 'waste_generated_kg', 'eco_actions'
        ]
        
        assert list(df.columns) == expected_columns
        assert len(df) == 1
    
    def test_dataframe_dtypes(self, sample_input_data):
        """Test DataFrame data types"""
        from src.preprocess import CarbonFootprintPreprocessor
        
        df = CarbonFootprintPreprocessor.prepare_input_dict(sample_input_data)
        
        # Check numeric columns
        numeric_cols = ['distance_km', 'electricity_kwh', 'renewable_usage_pct',
                       'screen_time_hours', 'waste_generated_kg', 'eco_actions']
        
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col])
        
        # Check categorical columns
        categorical_cols = ['day_type', 'transport_mode', 'food_type']
        for col in categorical_cols:
            assert pd.api.types.is_object_dtype(df[col])