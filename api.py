import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash-lite')

app = FastAPI(title="Form Autofill API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are a form autofill assistant. You receive:
1. parsed_data: JSON with form fields (selectors, labels, types, options)
   - Fields marked with "filled_by": "fuzzy_matching" are ALREADY FILLED - DO NOT TOUCH THEM
   - Only fill fields marked with "should_fill": true
2. personal_details: User's stored information

Your task:
- Match personal_details to ONLY EMPTY form fields (should_fill: true)
- SKIP any fields with "filled_by": "fuzzy_matching" 
- Return actions in this exact format:
{
  "actions": [
    {
      "selector": "#field_id",
      "action": "fill|check|uncheck|select|radio_select|select_multiple|fill_date|upload_file",
      "value": "matched_value",
      "confidence": 0.95,
      "reasoning": "Matched 'First Name' to firstName"
    }
  ],
  "summary": {
    "total_fields": 25,
    "already_filled": 15,
    "filled_by_ai": 8,
    "skipped": 2
  }
}

Action Types (USE APPROPRIATE TYPE):

1. fill - Text inputs (text, email, tel, number, url, textarea)
   {
     "action": "fill",
     "value": "John Doe"
   }

2. select - Single dropdown selection
   {
     "action": "select",
     "value": "option_value_or_text"
   }
   IMPORTANT: For select fields, use the option VALUE from parsed_data.options.unselected array (first item in [value, text] pair)

3. select_multiple - Multiple select dropdowns
   {
     "action": "select_multiple",
     "value": ["value1", "value2", "value3"]
   }
   Use for <select multiple> fields (skills, languages, certifications)

4. radio_select - Radio button groups
   {
     "action": "radio_select",
     "value": "option_value"
   }
   Select one option from a radio group (gender, employment type, visa status)

5. check - Single checkbox
   {
     "action": "check"
   }
   Use for: terms acceptance, single preferences, "yes/no" questions

6. uncheck - Uncheck checkbox
   {
     "action": "uncheck"
   }

7. fill_date - Date inputs with proper formatting
   {
     "action": "fill_date",
     "value": "18/11/2004"
   }
   Use date value from personal_details (DOB.day, DOB.month, DOB.year)
   Let the browser handle date formatting based on input type

8. upload_file - File upload fields
   {
     "action": "upload_file",
     "value": "Resume.pdf"
   }
   Use for: resume, CV, cover letter, portfolio uploads

9. spin_increment / spin_decrement - Number input adjustments
   {
     "action": "spin_increment"
   }
   Use for number inputs that need adjustment

CRITICAL RULES:
1. CHECK "filled_by" field - if it equals "fuzzy_matching", DO NOT CREATE ACTION for that field
2. ONLY process fields where "should_fill": true
3. NEVER alter selectors - use exact values from parsed_data.selector
4. For SELECT fields: ALWAYS use option VALUE from parsed_data.options.unselected
5. Handle nested personal_details (e.g., personal_details.DOB.month, personal_details.address.city)
6. Use confidence scores: 0.95 (exact match), 0.9 (high confidence), 0.85 (good match), 0.7+ (acceptable)
7. In summary, count fields by their status (already_filled, filled_by_ai, skipped)

MATCHING STRATEGY:
- Name fields: firstName, lastName, middleName, fullName
- Contact: email, phone, mobile, homePhone
- Address: Use address.streetAddress, address.city, address.state, address.postal, address.country
- Date fields: DOB.day, DOB.month, DOB.year or dateOfBirth arrays
- Education: education.degree, education.major, education.university, educationYear
- Employment: occupation, jobTitle, company, industry
- For dropdowns: match personal_details values to option values in parsed_data.options
- Gender: personal_details.gender → match to radio/select options
- Multiple selections: split comma-separated or use arrays from personal_details

IMPORTANT: Focus ONLY on fields that need filling (should_fill: true). Do not duplicate work already done by fuzzy matching.

RETURN ONLY VALID JSON - NO MARKDOWN, NO EXPLANATIONS"""


class AutofillRequest(BaseModel):
    parsed_data: Dict[str, Any]
    personal_details: Dict[str, Any]


class AutofillResponse(BaseModel):
    actions: list
    summary: Dict[str, int] = {}


def sanitize_selector(selector: str) -> str:
    """Sanitize CSS selectors for edge cases"""
    if not selector:
        return ""
    selector = selector.strip()
    
    # Handle invalid IDs like "#:r3:-form-item" → [id=':r3:-form-item']
    if selector.startswith("#:"):
        return f"[id='{selector[1:]}']"
    
    return selector


def call_llm(parsed_data: dict, personal_details: dict) -> dict:
    """Call Gemini to generate autofill actions"""
    
    # Count fields by status for logging
    all_fields = parsed_data.get('allFields', [])
    already_filled = sum(1 for f in all_fields if f.get('filled_by') == 'fuzzy_matching')
    should_fill = sum(1 for f in all_fields if f.get('should_fill') == True)
    
    print(f"📊 Form Analysis:")
    print(f"   - Total fields: {len(all_fields)}")
    print(f"   - Already filled by fuzzy: {already_filled}")
    print(f"   - Need AI filling: {should_fill}")
    
    prompt = f"""
{SYSTEM_PROMPT}

FORM DATA:
{json.dumps(parsed_data, indent=2)}

PERSONAL DETAILS:
{json.dumps(personal_details, indent=2)}

Generate autofill actions ONLY for empty fields (should_fill: true). Skip fields already filled by fuzzy matching:"""

    try:
        response = model.generate_content(prompt)
        result = response.text
        
        # Clean response (remove markdown wrappers)
        if result.startswith('```json'):
            result = result.split('```json')[1].split('```')[0].strip()
        elif result.startswith('```'):
            result = result.split('```')[1].split('```')[0].strip()
        
        # Parse JSON to validate
        actions = json.loads(result)

        # Sanitize selectors in actions
        if "actions" in actions:
            for act in actions["actions"]:
                act["selector"] = sanitize_selector(act.get("selector", ""))
        
        # Add summary if not present
        if "summary" not in actions:
            actions["summary"] = {
                "total_fields": len(all_fields),
                "already_filled": already_filled,
                "filled_by_ai": len(actions.get("actions", [])),
                "skipped": should_fill - len(actions.get("actions", []))
            }
        
        print(f"✅ AI Generated {len(actions.get('actions', []))} actions")
        
        return actions
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON Parse Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling LLM: {str(e)}")


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