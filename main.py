# main.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash-lite')

SYSTEM_PROMPT = """You are a form autofill assistant. You receive:
1. parsed_data: JSON with form fields (selectors, labels, types)
2. personal_details: User's stored information

Your task:
- Match personal_details to form fields using fuzzy matching
- Return actions in this exact format:
{
  "actions": [
    {
      "selector": "#field_id",
      "action": "fill|check|uncheck|select",
      "value": "matched_value",
      "confidence": 0.95,
      "reasoning": "Matched 'First Name' to FirstName"
    }
  ],
  "manual_fields": [
    {
      "selector": "#file_upload",
      "reason": "FILE_UPLOAD_REQUIRES_USER",
      "label": "Resume Upload"
    }
  ]
}

Action types:
- fill: text inputs (value: string)
- check: checkboxes/radio (value: true)
- uncheck: checkboxes (value: false)
- select: dropdowns (value: option value or text)

Rules:
- For select fields, use the option VALUE (e.g., "11" not "November")
- Match gender fields carefully to personal_details.Gender
- Only return actions for fields you can confidently match
- Return ONLY valid JSON, no explanations or markdown"""

# Hardcoded parsed data
parsed_data = {
    "url": "https://form.jotform.com/252470591559465",
    "title": "Online Internship Application Form",
    "sections": [
        {
            "heading": "Personal Information",
            "fields": [
                {
                    "selector": "#first_12",
                    "id": "first_12",
                    "type": "text",
                    "labels": {
                        "directLabel": "Full Name",
                        "contextText": "First Name"
                    },
                    "isRequired": False
                },
                {
                    "selector": "#last_12",
                    "id": "last_12",
                    "type": "text",
                    "labels": {
                        "directLabel": "Last Name",
                        "contextText": "Last Name"
                    },
                    "isRequired": False
                },
                {
                    "selector": "#input_54",
                    "id": "input_54",
                    "type": "text",
                    "labels": {
                        "directLabel": "Preferred Name"
                    },
                    "isRequired": False
                },
                {
                    "selector": "#input_20_0",
                    "id": "input_20_0",
                    "type": "checkbox",
                    "labels": {
                        "directLabel": "Male",
                        "groupLabel": "Gender",
                        "contextText": "Male"
                    },
                    "isRequired": False
                },
                {
                    "selector": "#input_20_1",
                    "id": "input_20_1",
                    "type": "checkbox",
                    "labels": {
                        "directLabel": "Female",
                        "groupLabel": "Gender",
                        "contextText": "Female"
                    },
                    "isRequired": False
                },
                {
                    "selector": "#input_19_month",
                    "id": "input_19_month",
                    "type": "select-one",
                    "labels": {
                        "directLabel": "Month",
                        "contextText": "Month"
                    },
                    "options": {
                        "unselected": [
                            ["", "Please select a month"],
                            ["1", "January"],
                            ["2", "February"],
                            ["3", "March"],
                            ["4", "April"],
                            ["5", "May"],
                            ["6", "June"],
                            ["7", "July"],
                            ["8", "August"],
                            ["9", "September"],
                            ["10", "October"],
                            ["11", "November"],
                            ["12", "December"]
                        ]
                    }
                },
                {
                    "selector": "#input_19_day",
                    "id": "input_19_day",
                    "type": "select-one",
                    "labels": {
                        "directLabel": "Day",
                        "contextText": "Day"
                    },
                    "options": {
                        "unselected": [
                            ["1", "1"],
                            ["2", "2"],
                            ["3", "3"],
                            ["15", "15"],
                            ["28", "28"]
                        ]
                    }
                },
                {
                    "selector": "#input_19_year",
                    "id": "input_19_year",
                    "type": "select-one",
                    "labels": {
                        "directLabel": "Year",
                        "contextText": "Year"
                    },
                    "options": {
                        "unselected": [
                            ["1990", "1990"],
                            ["1995", "1995"],
                            ["2000", "2000"],
                            ["2005", "2005"]
                        ]
                    }
                }
            ]
        }
    ]
}

# Hardcoded personal details
personal_details = {
    "FullName": "Jane Doe",
    "FirstName": "Jane",
    "LastName": "Doe",
    "PreferredName": "Jane",
    "Gender": "Female",
    "DateOfBirth": {
        "Month": "11",
        "Day": "15",
        "Year": "1995",
        "Full": "1995-11-15"
    },
    "Email": "jane.doe@example.com",
    "Phone": "+1-555-0123",
    "Address": {
        "Street": "123 Main St",
        "City": "San Francisco",
        "State": "CA",
        "ZipCode": "94101",
        "Country": "USA"
    }
}

def call_llm(parsed_data, personal_details):
    """Call Gemini to generate autofill actions"""
    
    prompt = f"""
{SYSTEM_PROMPT}

FORM DATA:
{json.dumps(parsed_data, indent=2)}

PERSONAL DETAILS:
{json.dumps(personal_details, indent=2)}

Generate autofill actions:"""

    try:
        response = model.generate_content(prompt)
        result = response.text
        
        # Clean response (remove markdown if present)
        if result.startswith('```json'):
            result = result.split('```json')[1].split('```')[0].strip()
        elif result.startswith('```'):
            result = result.split('```')[1].split('```')[0].strip()
        
        # Parse JSON to validate
        actions = json.loads(result)
        return actions
        
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

def main():
    print("=" * 60)
    print("AI FORM AUTOFILL - LLM CALL")
    print("=" * 60)
    
    print("\n📋 Parsed Form Data:")
    print(json.dumps(parsed_data, indent=2))
    
    print("\n👤 Personal Details:")
    print(json.dumps(personal_details, indent=2))
    
    print("\n🤖 Calling Gemini LLM...")
    actions = call_llm(parsed_data, personal_details)
    
    if actions:
        print("\n✅ Generated Actions:")
        print(json.dumps(actions, indent=2))
        
        # Save to file
        with open('output_actions.json', 'w') as f:
            json.dump(actions, f, indent=2)
        print("\n💾 Actions saved to output_actions.json")
    else:
        print("\n❌ Failed to generate actions")

if __name__ == '__main__':
    main()