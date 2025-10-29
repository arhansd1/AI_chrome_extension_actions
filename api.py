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
2. personal_details: User's stored information

Your task:
- Match personal_details to form fields using intelligent fuzzy matching
- Return actions in this exact format:
{
  "actions": [
    {
      "selector": "#field_id",
      "action": "fill|check|uncheck|select|radio_select|select_multiple|date_fill|upload_file|autocomplete|check_multiple",
      "value": "matched_value",
      "confidence": 0.95,
      "reasoning": "Matched 'First Name' to firstName"
    }
  ],
  "manual_fields": [
    {
      "selector": "#complex_field",
      "reason": "REQUIRES_MANUAL_INPUT",
      "label": "Field Label"
    }
  ]
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
     "value": "option_value",
     "groupName": "field_name"
   }
   Select one option from a radio group (gender, employment type, visa status)

5. check - Single checkbox or radio button
   {
     "action": "check",
     "value": true
   }
   Use for: terms acceptance, single preferences, "yes/no" questions

6. uncheck - Uncheck checkbox
   {
     "action": "uncheck",
     "value": false
   }

7. check_multiple - Multiple checkboxes in same group
   {
     "action": "check_multiple",
     "selectors": ["#check1", "#check2"],
     "values": ["value1", "value2"]
   }
   Use for: interests, preferences, multiple certifications

8. date_fill - Date inputs with proper formatting
   {
     "action": "date_fill",
     "value": "18/11/2004",
     "format": "DD/MM/YYYY"
   }
   Detect format from field attributes or common patterns
   Supported formats: "DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD"

9. upload_file - File upload fields
   {
     "action": "upload_file",
     "value": "Resume.pdf",
     "fileType": "resume"
   }
   Use for: resume, CV, cover letter, portfolio uploads

10. autocomplete - Autocomplete/combobox fields
    {
      "action": "autocomplete",
      "value": "San Francisco",
      "typingDelay": 100
    }
    Use for: location, company, university fields with search

CRITICAL RULES:
1. NEVER alter selectors - use exact values from parsed_data.selector (keep ':', special chars)
2. For SELECT fields: ALWAYS use option VALUE (e.g., "11" not "November") from parsed_data.options.unselected
3. Match date formats to field requirements (check parsed_data.validation.pattern or common formats)
4. For radio buttons: identify the group name and select appropriate value
5. For checkboxes: determine if single (check/uncheck) or multiple (check_multiple)
6. File uploads: match to resume/CV/document fields
7. Handle nested personal_details (e.g., personal_details.DOB.month, personal_details.address.city)
8. Use confidence scores: 0.95 (exact match), 0.9 (high confidence), 0.85 (good match), 0.7+ (acceptable)

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

RETURN ONLY VALID JSON - NO MARKDOWN, NO EXPLANATIONS"""


class AutofillRequest(BaseModel):
    parsed_data: Dict[str, Any]
    personal_details: Dict[str, Any]


class AutofillResponse(BaseModel):
    actions: list
    manual_fields: list = []


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
    
    prompt = f"""
{SYSTEM_PROMPT}

FORM DATA:
{json.dumps(parsed_data, indent=2)}

PERSONAL DETAILS:
{json.dumps(personal_details, indent=2)}

Generate comprehensive autofill actions for ALL matchable fields:"""

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
                
        if "manual_fields" in actions:
            for field in actions["manual_fields"]:
                field["selector"] = sanitize_selector(field.get("selector", ""))

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
    
    - **parsed_data**: The parsed form structure
    - **personal_details**: User's personal information
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
    uvicorn.run(app, host="0.0.0.0", port=8000)