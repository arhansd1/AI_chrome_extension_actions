import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
from fastapi import HTTPException

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash-lite')

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
      "confidence": 'Numeric value between 0-1 indicating match certainty (0.9+ = exact, 0.7-0.89 = high, 0.5-0.69 = medium, <0.5 = low)',
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
     "value": "2004-11-18"  # Use YYYY-MM-DD format for date inputs
   }
   Convert from personal_details date format to ISO format (YYYY-MM-DD)
   Use date value from personal_details (DOB.day, DOB.month, DOB.year)

8. upload_file - File upload fields
   {
     "action": "upload_file",
     "value": "filename_from_personal_details.resume_filename"
   }
   Use for: resume, CV, cover letter, portfolio uploads
   IMPORTANT: Check if personal_details contains 'resume_base64' and 'resume_filename'
    Only create upload_file action if both exist and are not null

9. spin_increment / spin_decrement - Number input adjustments
   {
     "action": "spin_increment"
   }
   Use for number inputs that need adjustment

CRITICAL RULES:
1. CHECK "filled_by" field - if it equals "fuzzy_matching", DO NOT CREATE ACTION for that field
2. ONLY process fields where "should_fill": true
3. If value in a certain field is 'null' or 'string' do not use that to fill the field as its a error.
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
- Education arrays: personal_details.educations[0].institution, .degree, .field_of_study, .start_year, .end_year
- Experience arrays: personal_details.experiences[0].company, .title, .start_month, .start_year, .description
- Handle multiple education/work entries by matching to corresponding form sections
- For dropdowns: match personal_details values to option values in parsed_data.options
- Gender: personal_details.gender → match to radio/select options
- Multiple selections: split comma-separated or use arrays from personal_details

RETURN ONLY VALID JSON - NO MARKDOWN, NO EXPLANATIONS"""


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