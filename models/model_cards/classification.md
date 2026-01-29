# Model Card: Carbon Impact Level Classification Model

## Model Details
- **Model Name**: Carbon Impact Level Classification Model
- **Version**: 1.0.0
- **Type**: Classification (Random Forest)
- **Purpose**: Classify daily carbon impact as Low/Medium/High

## Intended Use
- **Primary Use**: Categorize carbon footprint impact levels
- **Users**: Environmental educators, personal tracking apps
- **Limitations**: Threshold-based classification may not suit all contexts

## Training Data
- **Dataset**: Personal Carbon Footprint Behavior Dataset
- **Samples**: 1,400 daily records
- **Features**: 9 behavioral features
- **Target**: `carbon_impact_level` (Low/Medium/High)
- **Distribution**: Low (27.6%), Medium (49.4%), High (23.0%)

## Performance Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| Accuracy | 81.4% | Overall correct predictions |
| Precision (Macro) | 81.2% | Average precision across classes |
| Recall (Macro) | 81.4% | Average recall across classes |
| F1-Score (Macro) | 81.1% | Harmonic mean of precision/recall |
| ROC AUC (Macro) | 0.89 | Area under ROC curve |

## Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low | 0.82 | 0.85 | 0.83 | 77 |
| Medium | 0.81 | 0.80 | 0.80 | 138 |
| High | 0.82 | 0.83 | 0.82 | 65 |

## Classification Thresholds
- **Low**: Carbon footprint < 6.5 kg
- **Medium**: 6.5 kg ≤ Footprint ≤ 10.5 kg
- **High**: Carbon footprint > 10.5 kg

## Feature Importance
1. **Electricity Usage (20%)**: Primary predictor
2. **Renewable Energy % (18%)**: Key sustainability indicator
3. **Transport Distance (16%)**: Travel impact
4. **Car Transport (12%)**: High-emission transport mode
5. **Waste Generated (9%)**: Waste management impact

## Ethical Considerations
- **Transparency**: Thresholds are data-driven but adjustable
- **Fairness**: Classes based on dataset distribution
- **Actionability**: Provides clear categories for intervention

## Recommendations
1. Use as educational tool for awareness
2. Combine with regression predictions
3. Allow threshold customization for different contexts
4. Update thresholds with new sustainability standards

## Contact
For questions or issues, contact: models@carbonfootprint.ai