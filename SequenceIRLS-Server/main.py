from SequenceIRLS import complete_sequence, hm_irls_iterative
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union, Dict
import numpy as np
import asyncio
import json
from fastapi.responses import StreamingResponse
import time


app = FastAPI(title="Sequence IRLS", version="1.0.0")

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SequenceSettings(BaseModel):
    tolerance: float = 1e-6
    rank: int = 10
    max_iters: int = 1000
    mean_type: str = "harmonic"  # "geometric" or "harmonic"

class SequenceRequest(BaseModel):
    sequence: List[Optional[float]]  # None for missing values
    settings: SequenceSettings = SequenceSettings()


class PredictionUpdate(BaseModel):
    sigma: List[float] 
    elements_updated: List[float]
    error: float
    iteration: int

class SequenceResponse(BaseModel):
    elements_updated: List[float]
    sigmas: List[float]
    error: float


def parse_sequence_settings(request: SequenceRequest):
    """
    class SequenceRequest(BaseModel):
    sequence: List[Optional[float]]  # None for missing values
    settings: SequenceSettings = SequenceSettings()
    """
    if not request.sequence:
            raise HTTPException(status_code=400, detail="Sequence cannot be empty")
        
        # Check if there are any missing values
    missing_indices = [i for i, val in enumerate(request.sequence) if val is None]
    if not missing_indices:
        raise HTTPException(status_code=400, detail="No missing values to predict")
    
    entries = len(request.sequence)-len(missing_indices)
    
    if not 0 < request.settings.rank < len(request.sequence):
        raise HTTPException(status_code=400, detail="Rank invalid, should be the order of the sequence")
    
    if entries < request.settings.rank:
        raise HTTPException(status_code=400, detail="No enough entries given the provided rank")
    
    if not 0 < request.settings.max_iters <=10000:
        raise HTTPException(status_code=400, detail="Max iterations invalid should 0-10000")
    
    if not 0 < request.settings.tolerance <= 10**-2:
        raise HTTPException(status_code=400, detail="Tolerance invalid should be between 0 and 0.1, recommended 10**-15")    


@app.get("/")
async def root():
    return {"message": "Hello!  Sequence Completion API", "status": "running"}


@app.post("/predict")
async def predict_sequence(request: SequenceRequest):
    """Non-streaming prediction endpoint"""
        
    sequence = request.sequence
    settings = request.settings

    try:
        predicted_sequence, sigmas, error = complete_sequence(
            data_vector=sequence,
            rank_estimate=settings.rank,
            max_iter=settings.max_iters,
            tol=settings.tolerance,
            type_mean=settings.mean_type
        ) 
    except IndexError:
        raise HTTPException(status_code=400, detail="Given the current rank, there needs to be more observed data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error")
    
    


    return SequenceResponse(
            elements_updated=predicted_sequence,
            sigmas=sigmas,
            error=error
        )
    




from fastapi.responses import StreamingResponse
import asyncio
import json

@app.post("/predict/stream")
async def predict_sequence_stream(request: SequenceRequest):
    """Streaming prediction endpoint for real-time updates"""
    try:
        if not request.sequence:
            raise HTTPException(status_code=400, detail="Sequence cannot be empty")
        
        missing_indices = [i for i, val in enumerate(request.sequence) if val is None]
        if not missing_indices:
            raise HTTPException(status_code=400, detail="No missing values to predict")
        
        
        sequence = request.sequence
        settings = request.settings
        print(settings)

        async def generate_predictions():
            idx = 0
            async for S, x, err in hm_irls_iterative(
                data_vector=sequence,
                rank_estimate=settings.rank,
                max_iter=settings.max_iters,
                tol=settings.tolerance,
                type_mean=settings.mean_type
            ):
                # Convert predictions to your model
                pred_data = PredictionUpdate(
                    iteration=idx,
                    elements_updated=x,  # you may need to convert x if it's not already a list of PredictedElement
                    sigma=[float(s) for s in S],
                    error=float(err),
                )

                data = {
                    "type": "prediction_update",
                    "data": pred_data.dict()  # convert Pydantic object to dict
                }

                yield f"data: {json.dumps(data)}\n\n"
                idx += 1

            # Send completion signal
            completion_data = {
                "type": "completion",
                "data": {"completed": True}
            }
            yield f"data: {json.dumps(completion_data)}\n\n"

        return StreamingResponse(
            generate_predictions(),
            media_type="text/event-stream",  # better for streaming events
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")


@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
