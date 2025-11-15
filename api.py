from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from agent import call_llm

app = FastAPI(title="Form Autofill API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AutofillRequest(BaseModel):
    parsed_data: Dict[str, Any]
    personal_details: Dict[str, Any]


class AutofillResponse(BaseModel):
    actions: list
    summary: Dict[str, int] = {}


@app.get("/")
def read_root():
    return {
        "message": "Form Autofill API",
        "version": "1.0.0",
        "endpoints": {
            "/autofill": "POST - Generate autofill actions",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/autofill", response_model=AutofillResponse)
def generate_autofill(request: AutofillRequest):
    """
    Generate autofill actions based on parsed form data and personal details
    
    - **parsed_data**: The parsed form structure (with filled_by markers from fuzzy matching)
    - **personal_details**: User's personal information
    
    Returns actions ONLY for fields not already filled by fuzzy matching
    """
    try:
        actions = call_llm(request.parsed_data, request.personal_details)
        return actions
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8070)