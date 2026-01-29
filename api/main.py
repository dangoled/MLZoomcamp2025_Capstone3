"""
FastAPI application for Carbon Footprint Prediction
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import pickle
from typing import List, Optional, Dict, Any
import os
from datetime import datetime
import json

from .schemas import (
    CarbonFootprintInput, 
    CarbonFootprintResponse,
    BatchPredictionRequest,
    ModelInfoResponse,
    PredictionExplanation
)
from .predict import CarbonFootprintPredictor

app = FastAPI(
    title="Carbon Footprint Prediction API",
    description="API for predicting personal carbon footprint and impact level",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
MODELS_DIR = os.getenv('MODELS_DIR', 'models')
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global predictor
    try:
        predictor = CarbonFootprintPredictor(MODELS_DIR)
        print(f"Models loaded successfully from {MODELS_DIR}")
        print(f"Available models: {list(predictor.models.keys())}")
    except Exception as e:
        print(f"Error loading models: {e}")
        predictor = None

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "Carbon Footprint Prediction API",
        "status": "operational" if predictor else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "predict-batch": "/predict-batch",
            "model-info": "/model-info",
            "explain": "/explain/{prediction_id}"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "status": "healthy",
        "models_loaded": True,
        "model_count": len(predictor.models) if predictor else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=CarbonFootprintResponse, tags=["Prediction"])
async def predict_single(input_data: CarbonFootprintInput):
    """
    Predict carbon footprint for a single day's activities
    
    - **input_data**: Daily activity data including transport, energy usage, etc.
    - Returns: Predicted carbon footprint and impact level with explanations
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Make prediction
        result = predictor.predict(input_data.dict())
        
        # Generate prediction ID
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(input_data.dict())) % 10000:04d}"
        
        return CarbonFootprintResponse(
            prediction_id=prediction_id,
            carbon_footprint_kg=result['carbon_footprint_kg'],
            carbon_impact_level=result['carbon_impact_level'],
            confidence=result.get('confidence', 0.0),
            suggestions=result.get('suggestions', []),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-batch", response_model=List[CarbonFootprintResponse], tags=["Prediction"])
async def predict_batch(batch_request: BatchPredictionRequest):
    """
    Predict carbon footprint for multiple days
    
    - **batch_request**: List of daily activity data
    - Returns: List of predictions
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        results = []
        
        for i, input_data in enumerate(batch_request.activities):
            result = predictor.predict(input_data.dict())
            
            prediction_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:04d}"
            
            response = CarbonFootprintResponse(
                prediction_id=prediction_id,
                carbon_footprint_kg=result['carbon_footprint_kg'],
                carbon_impact_level=result['carbon_impact_level'],
                confidence=result.get('confidence', 0.0),
                suggestions=result.get('suggestions', []),
                timestamp=datetime.now().isoformat()
            )
            
            results.append(response)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Models"])
async def model_info(
    model_type: Optional[str] = Query(None, description="Filter by model type (regression/classification)")
):
    """
    Get information about loaded models
    
    - **model_type**: Optional filter for model type
    - Returns: Model information including performance metrics
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        return predictor.get_model_info(model_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explain/{prediction_id}", response_model=PredictionExplanation, tags=["Explanation"])
async def explain_prediction(
    prediction_id: str,
    input_data: Optional[CarbonFootprintInput] = None
):
    """
    Get explanation for a prediction
    
    - **prediction_id**: ID of the prediction to explain
    - **input_data**: Optional input data (if not provided, uses sample)
    - Returns: Detailed explanation of the prediction
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        if input_data is None:
            # Use sample input for demonstration
            from src.preprocess import CarbonFootprintPreprocessor
            sample_input = CarbonFootprintPreprocessor.get_sample_input()
            input_data = CarbonFootprintInput(**sample_input)
        
        result = predictor.predict(input_data.dict())
        explanation = predictor.explain_prediction(input_data.dict())
        
        return PredictionExplanation(
            prediction_id=prediction_id,
            input_data=input_data.dict(),
            prediction=result,
            explanation=explanation,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/sample-input", tags=["Examples"])
async def get_sample_input():
    """Get sample input for testing the API"""
    from src.preprocess import CarbonFootprintPreprocessor
    sample = CarbonFootprintPreprocessor.get_sample_input()
    
    return {
        "sample_input": sample,
        "description": "Sample input data for testing predictions",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)