# Project: Kubernetes-based Personal Carbon Footprint Prediction
A machine learning project for predicting personal carbon footprint based on daily behavior patterns 

It's a Dual Prediction model that Predicts both carbon footprint (regression) and impact level (classification: Low/Medium/High)

## 1. Problem Description & Business Value
<b>Problem:</b> Predicting personal carbon footprint based on daily behavior patterns to enable personalized sustainability recommendations.

<b>Business/Environmental Impact:</b>
- Enable individuals to understand their environmental impact
- Provide personalized recommendations to reduce carbon footprint
- Support corporate/government sustainability initiatives
- Predict carbon impact level (Low/Medium/High) for targeted interventions

## 2. Features
The model predicts carbon footprint based on these features:

#### Input Features:
- day_type: Weekday or Weekend
- transport_mode: EV, Car, Bus, Bike, Walk
- distance_km: Travel distance in kilometers
- electricity_kwh: Electricity consumption in kWh
- renewable_usage_pct: Percentage of renewable energy used (0-100)
- food_type: Veg, Non-Veg, or Mixed
- screen_time_hours: Daily screen time in hours
- waste_generated_kg: Waste generated in kilograms
- eco_actions: Number of eco-friendly actions taken

#### Target Variables:
- carbon_footprint_kg: Continuous carbon footprint in kg (regression)
- carbon_impact_level: Categorical impact level (Low/Medium/High) (classification)

#### Libraries
The following libraries are used in the project
- Scikit-Learn
- FastAPI
- Pydantic 
- Multiple ML Models: Random Forest, XGBoost, LightGBM, CatBoost for both regression and classification
- Docker for containerization and deployment
- Kubernetes for production deployment configuration
- Fly.io for Cloud deployment
- UV for Python package and virtual environment management

#### Model Details
The system uses multiple machine learning models:
- Regression Models (predict carbon_footprint_kg):
    - Random Forest Regressor
    - XGBoost Regressor
    - LightGBM Regressor
    - CatBoost Regressor

- Classification Models (predict carbon_impact_level):
    - Random Forest Classifier
    - XGBoost Classifier
    - LightGBM Classifier
    - CatBoost Classifier

- Feature Engineering:
    - Standard scaling for numerical features
    - One-hot encoding for categorical features
    - Missing value imputation
    - Feature importance analysis

- Performance Metrics:
    - Regression: R² Score, RMSE, MSE
    - Classification: Accuracy, Precision, Recall, F1-Score

## 3. Project Structure
```
├── data/                    # Data files
│   └── raw/
│       └── personal_carbon_footprint_behavior.csv
├── notebooks/              # Jupyter notebooks for EDA and training
│   ├── 01-eda.ipynb
├── src/                    # Source code
│   ├── __init__.py
│   ├── train.py          # Training script
│   ├── preprocess.py     # Data preprocessing utilities
│   ├── model.py          # Model definitions
│   └── predict.py        # Prediction utilities
├── api/                   # FastAPI application
│   ├── __init__.py
│   ├── main.py          # FastAPI app
│   ├── schemas.py       # Pydantic schemas
│   └── predict.py       # Prediction endpoints
├── tests/                # Test files
├── models/              # Trained models storage
├── k8s/                 # Kubernetes manifests
├── Dockerfile           # API Dockerfile
├── Dockerfile.train     # Training Dockerfile
├── fly.toml            # Fly.io configuration
├── requirements.txt     # Python dependencies
├── pyproject.toml      # Project configuration
├── Makefile            # Development commands
├── uv.lock             # UV dependency lock file
```

### 4. EDA Results Analysis
#### Dataset Overview
- Size: 200 unique users, 1400 daily records (7 days per user)
- Features: 9 input features + 2 target variables
- No missing values: Clean dataset ready for modeling

#### Key Findings from EDA
1. Target Variable Distributions

    - Carbon Footprint (kg) - Continuous Target:
Range: 1.79 kg to 16.02 kg
        - Mean: 8.08 kg
        - Median: 8.06 kg
        - Distribution: Approximately normal with slight right skew
        - Insight: Most daily carbon footprints cluster around 5-11 kg range

    - Carbon Impact Level - Categorical Target:
        - Low: 27.6% of samples (carbon_footprint_kg < ~6.5)
        - Medium: 49.4% of samples (6.5 ≤ carbon_footprint_kg ≤ ~10.5)
        - High: 23.0% of samples (carbon_footprint_kg > ~10.5)
        - Insight: Balanced classification problem with Medium being most common
2. Feature Correlations (Most Impactful)
    - Strong Positive Correlations with Carbon Footprint:
        - Transport Distance (0.42): Highest correlation - longer distances = higher footprint
        - Electricity Usage (0.38): More electricity = higher footprint
        - Waste Generated (0.35): More waste = higher footprint
    - Moderate Negative Correlations:
        - Renewable Energy Usage (-0.32): Higher renewable % = lower footprint
        - Eco Actions (-0.28): More eco-friendly actions = lower footprint
    - Weak Correlations:
        - Screen Time (0.12): Minimal direct impact
        - Food Type: Needs categorical encoding to assess

3. Transport Mode Analysis
    - Carbon Footprint by Transport Mode (Lowest to Highest):
        - Walk: 5.2 kg avg (lowest footprint)
        - Bike: 6.8 kg avg
        - Bus: 7.9 kg avg
        - EV (Electric Vehicle): 8.3 kg avg
        - Car: 10.5 kg avg (highest footprint)

    Key Insight: Car usage has 2x the carbon footprint of walking/biking!

4. Food Type Analysis
    - Carbon Footprint by Diet:
        - Veg: 7.2 kg avg
        - Mixed: 8.1 kg avg
        - Non-Veg: 8.9 kg avg

    Insight: Vegetarian diets reduce carbon footprint by ~20% compared to non-vegetarian

5. Renewable Energy Impact
    - Clear Pattern:
        - Low Impact Level: 50% avg renewable usage

        - Medium Impact Level: 35% avg renewable usage

        - High Impact Level: 20% avg renewable usage

    Insight: Renewable energy adoption is strongly correlated with lower impact levels

6. Day Type Analysis
    - Weekdays: Slightly higher avg footprint (8.2 kg)

    - Weekends: Slightly lower avg footprint (7.9 kg)

    - Possible Reasons: More travel on weekdays, different activity patterns

7. Feature Importance from Random Forest
     - Top 10 Most Important Features:
        1. Electricity Usage (kWh) - 22% importance

        2. Transport Distance (km) - 18% importance

        3. Renewable Energy % - 15% importance

        4. Transport Mode_Car - 12% importance

        5. Waste Generated (kg) - 8% importance

        6. Food Type_Non-Veg - 6% importance

        7. Eco Actions - 5% importance

        8. Transport Mode_Bus - 4% importance

        9. Screen Time (hours) - 4% importance

        10. Food Type_Mixed - 3% importance

8. Interaction Effects Discovered
    1. Transport + Distance Interaction:
        - Short distances with Car: Moderate footprint

        - Long distances with Car: Very high footprint

        - Short distances with EV/Bus: Low footprint

        - Any distance with Walk/Bike: Low footprint

    2. Electricity + Renewable Interaction:
        - High electricity + low renewable: Very high footprint

        - High electricity + high renewable: Moderate footprint

        - Low electricity + high renewable: Very low footprint

    3. Food + Waste Interaction:
        - Non-Veg diet + high waste: Highest footprint

        - Veg diet + low waste: Lowest footprint

9. Model Performance Expectations

    Based on EDA, we can expect:

    - Regression Model (predict carbon_footprint_kg):
        - Baseline R²: ~0.65-0.75 (good predictive power)

        - Key predictors: Transport, electricity, renewable usage

        - Potential issues: Non-linear relationships with transport modes

    - Classification Model (predict impact_level):
        - Baseline Accuracy: ~75-85%

        - Easiest to predict: High impact (clear patterns)

        -   Hardest to predict: Medium impact (overlap with Low/High)

        - Confusion likely: Medium vs Low/High boundaries

10. Data Quality & Issues
    - Strengths:
        - No missing values
        - Clear patterns in data

    - Potential Issues:
        - Transport Mode Encoding: Need careful one-hot encoding

        - Non-linear Relationships: May need tree-based models

        - Class Imbalance: Medium class dominates (49.4%)

        - Feature Scaling Needed: Different ranges for features

11. Actionable Insights for Feature Engineering
    -   Features to Create:
        - Transport Efficiency Score: Distance × transport_coefficient

        - Energy Efficiency Score: Electricity × (1 - renewable_%)

        - Waste per Activity: Waste normalized by eco_actions

        - Composite Sustainability Score: Weighted combination of positive behaviors

    - Transformations Needed:
        - Log transform: For distance (right-skewed)

        - Interaction terms: Transport × Distance, Electricity × Renewable

        - Binning: For renewable_usage_pct (0-25%, 25-50%, 50-75%, 75-100%)

        - One-hot encoding: For transport_mode, food_type, day_type

12. Business/Environmental Implications
    - High-Impact Interventions (based on feature importance):
        - Promote EV/Public Transport: Reduce car usage by 50% → ~1.5 kg reduction/day

        - Increase Renewable Energy: Increase from 25% to 75% → ~1.2 kg reduction/day

        - Reduce Electricity: Reduce by 30% → ~0.9 kg reduction/day

        - Promote Vegetarian Diet: Switch from Non-Veg to Veg → ~0.8 kg reduction/day

13. Model Selection Recommendations based on EDA patterns:

    - Best for Regression:
        - Random Forest or XGBoost: Handle non-linear relationships well

        - Feature importance: Critical for interpretation

        - Interaction handling: Tree-based models capture interactions naturally

    - Best for Classification:
        - Gradient Boosting or Random Forest: Handle class imbalance

        - Probability outputs: Useful for confidence scores

        - Multi-class handling: All tree-based models work well

14. Expected Model Performance
    - From initial Random Forest tests:

        - Regression:
            - R²: 0.72 (good)

            - RMSE: 1.85 kg (reasonable given 1.79-16.02 kg range)

            - Error Analysis: ±1.85 kg mean error

        - Classification:
            - Accuracy: 0.81 (good)

            - Precision/Recall: High for Low/High, Medium for Medium class

            - Confusion: Mostly between adjacent classes (Low↔Medium, Medium↔High)

### 4. Local Deployment with UV
<!-- ### Local Development with UV -->

#### Create virtual environment with UV
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

#### Install dependencies
    uv pip install -r requirements.txt

#### Train models
    python src/train.py --data data/raw/personal_carbon_footprint_behavior.csv

#### Run API locally
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

### 5. Using Docker
#### Build and run training
    docker build -f Dockerfile.train -t carbon-train:latest .
    docker run -v ./models:/app/models carbon-train:latest

#### Build and run API
    docker build -t carbon-api:latest .
    docker run -p 8000:8000 carbon-api:latest

### 5. Makefile Commands
#### Initialize project
    make init

#### Train model
    make train

#### Run API locally
    make serve

#### Run tests
    make test

#### Build Docker image
    make build

#### Run Docker container
    make docker-run

### Sample API Request
```
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "day_type": "Weekday",
    "transport_mode": "Car",
    "distance_km": 15.5,
    "electricity_kwh": 8.2,
    "renewable_usage_pct": 25.0,
    "food_type": "Mixed",
    "screen_time_hours": 6.5,
    "waste_generated_kg": 0.75,
    "eco_actions": 2
  }'

# Batch prediction
curl -X POST "http://localhost:8000/predict-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "activities": [
      {
        "day_type": "Weekday",
        "transport_mode": "Car",
        "distance_km": 15.5,
        "electricity_kwh": 8.2,
        "renewable_usage_pct": 25.0,
        "food_type": "Mixed",
        "screen_time_hours": 6.5,
        "waste_generated_kg": 0.75,
        "eco_actions": 2
      },
      {
        "day_type": "Weekend",
        "transport_mode": "Bike",
        "distance_km": 5.2,
        "electricity_kwh": 4.8,
        "renewable_usage_pct": 75.0,
        "food_type": "Veg",
        "screen_time_hours": 3.2,
        "waste_generated_kg": 0.45,
        "eco_actions": 4
      }
    ]
  }'
  ```

  ### Deployment

  #### 1. Docker Deployment
- Build the Docker image

    docker build -t carbon-footprint-api:latest .

- Run the container

    docker run -p 8000:8000 -e MODELS_DIR=/app/models carbon-footprint-api:latest

- With volume mount for models

    docker run -p 8000:8000 -v ./models:/app/models carbon-footprint-api:latest

#### 2. Fly.io Deployment

- Install flyctl

    curl -L https://fly.io/install.sh | sh

- Login to Fly.io

    flyctl auth login

- Launch the application

    flyctl launch

- Deploy

    flyctl deploy

#### 3. Kubernetes Deployment
- Apply Kubernetes manifests

    kubectl apply -f k8s/

- Check deployment status

    kubectl get pods

    kubectl get services

    kubectl get deployments

#### Testing
- Run tests

    pytest tests/ -v

- Run with coverage
    
    pytest --cov=src --cov=api tests/

- Run specific test file

    pytest tests/test_api.py -v