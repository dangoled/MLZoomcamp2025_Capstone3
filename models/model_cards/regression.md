# Model Card: Carbon Footprint Regression Model

## Model Details
- **Model Name**: Carbon Footprint Regression Model
- **Version**: 1.0.0
- **Type**: Regression (Random Forest)
- **Purpose**: Predict daily carbon footprint in kg CO₂

## Intended Use
- **Primary Use**: Estimate personal carbon footprint based on daily activities
- **Users**: Individuals, sustainability apps, carbon tracking platforms
- **Limitations**: Trained on specific dataset; may not generalize to all populations

## Training Data
- **Dataset**: Personal Carbon Footprint Behavior Dataset
- **Samples**: 1,400 daily records from 200 users
- **Features**: 9 behavioral features
- **Target**: `carbon_footprint_kg` (continuous, 1.79-16.02 kg range)
- **Split**: 80% training, 20% testing

## Performance Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| R² Score | 0.724 | Proportion of variance explained |
| RMSE | 1.845 kg | Root mean squared error |
| MSE | 3.402 | Mean squared error |
| Mean Absolute Error | 1.412 kg | Average absolute error |
| Within ±1kg | 67.5% | Predictions within 1 kg of actual |
| Within ±2kg | 92.3% | Predictions within 2 kg of actual |

## Feature Importance
1. **Electricity Usage (22%)**: kWh of electricity consumed
2. **Transport Distance (18%)**: Distance traveled in km
3. **Renewable Energy % (15%)**: Percentage of renewable energy used
4. **Car Transport (12%)**: Using car as transport mode
5. **Waste Generated (8%)**: kg of waste produced

## Ethical Considerations
- **Bias**: Model trained on limited demographic data
- **Fairness**: Should be validated across diverse populations
- **Privacy**: Uses only behavioral data, no personal identifiers

## Recommendations
1. Use predictions as estimates, not exact measurements
2. Combine with other sustainability metrics
3. Regularly update with new data
4. Provide confidence intervals with predictions

## Contact
For questions or issues, contact: models@carbonfootprint.ai