import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re

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
- When referencing an element by ID, always use the exact value from parsed_data.selector.
- Never alter or sanitize selectors (keep ':' or special characters as-is).
- For select fields, use the option VALUE (e.g., "11" not "November")
- Match gender fields carefully to personal_details.Gender
- Only return actions for fields you can confidently match
- Return ONLY valid JSON, no explanations or markdown"""

# Hardcoded parsed data
parsed_data_json = r'''
{
  "url": "https://job.10xscale.ai/4846461985313787904",
  "title": "Hire 10x Application form",
  "timestamp": "2025-10-24T11:50:52.202Z",
  "sections": [
    {
      "heading": "Fill Your Application",
      "fields": [
        {
          "selector": "#file-input",
          "id": "file-input",
          "name": "",
          "type": "file",
          "inputType": "file",
          "labels": {
            "placeholder": "",
            "precedingLabels": [
              "Fill Your Application"
            ],
            "contextText": "Upload your resume or drag and drop it hereOnly .doc, .docx, .pdf"
          },
          "value": null,
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "[name=\"name.firstName\"]",
          "id": "",
          "name": "name.firstName",
          "type": "text",
          "inputType": "text",
          "labels": {
            "placeholder": "First Name",
            "precedingLabels": [
              "Fill Your Application",
              "Name*"
            ]
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "[name=\"name.lastName\"]",
          "id": "",
          "name": "name.lastName",
          "type": "text",
          "inputType": "text",
          "labels": {
            "placeholder": "Last Name",
            "precedingLabels": [
              "Fill Your Application",
              "Name*"
            ]
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "input.flex.h-10.w-full.rounded-md.border.border-input.bg-background.py-2.ring-offset-background.file:border-0.file:bg-transparent.file:text-sm.file:font-medium.placeholder:text-muted-foreground.focus-visible:outline-none.focus-visible:ring-2.focus-visible:ring-ring.focus-visible:ring-offset-2.disabled:cursor-not-allowed.disabled:opacity-50.flex-1.min-w-[200px].border-none.outline-none.text-sm.placeholder-gray-400.focus:ring-0.px-2:nth-child(1)",
          "id": "",
          "name": "",
          "type": "text",
          "inputType": "text",
          "labels": {
            "placeholder": "Add email address...",
            "precedingLabels": [
              "Fill Your Application",
              "Name*",
              "Email*"
            ]
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "input.form-control:nth-child(2)",
          "id": "",
          "name": "",
          "type": "tel",
          "inputType": "tel",
          "labels": {
            "placeholder": "Enter Contact Number",
            "contextText": "Phone"
          },
          "value": "+91",
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#:r3:-form-item",
          "id": ":r3:-form-item",
          "name": "address",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Address*",
            "placeholder": "Enter Current Address",
            "precedingLabels": [
              "Name*",
              "Email*",
              "Phone*",
              "Phone",
              "Address*"
            ],
            "contextText": "Address*"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#:r4:-form-item",
          "id": ":r4:-form-item",
          "name": "prefLocation",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Preferred Location",
            "placeholder": "Enter Preferences",
            "precedingLabels": [
              "Email*",
              "Phone*",
              "Phone",
              "Address*",
              "Preferred Location"
            ],
            "contextText": "Preferred Location"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#:r5:-form-item",
          "id": ":r5:-form-item",
          "name": "current_ctc",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Current Salary*",
            "placeholder": "Enter Current Salary",
            "precedingLabels": [
              "Phone*",
              "Phone",
              "Address*",
              "Preferred Location",
              "Current Salary*"
            ],
            "contextText": "Current Salary*"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#:r6:-form-item",
          "id": ":r6:-form-item",
          "name": "expected_ctc",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Expected Salary*",
            "placeholder": "Enter Expected Salary",
            "precedingLabels": [
              "Phone",
              "Address*",
              "Preferred Location",
              "Current Salary*",
              "Expected Salary*"
            ],
            "contextText": "Expected Salary*"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#:r7:-form-item",
          "id": ":r7:-form-item",
          "name": "noticePeriod",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Notice Period*",
            "placeholder": "Enter Notice Period",
            "precedingLabels": [
              "Address*",
              "Preferred Location",
              "Current Salary*",
              "Expected Salary*",
              "Notice Period*"
            ],
            "contextText": "Notice Period*"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        }
      ],
      "subsections": []
    }
  ],
  "allFields": [
    {
      "selector": "#file-input",
      "id": "file-input",
      "name": "",
      "type": "file",
      "inputType": "file",
      "labels": {
        "placeholder": "",
        "precedingLabels": [
          "Fill Your Application"
        ],
        "contextText": "Upload your resume or drag and drop it hereOnly .doc, .docx, .pdf"
      },
      "value": null,
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "[name=\"name.firstName\"]",
      "id": "",
      "name": "name.firstName",
      "type": "text",
      "inputType": "text",
      "labels": {
        "placeholder": "First Name",
        "precedingLabels": [
          "Fill Your Application",
          "Name*"
        ]
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "[name=\"name.lastName\"]",
      "id": "",
      "name": "name.lastName",
      "type": "text",
      "inputType": "text",
      "labels": {
        "placeholder": "Last Name",
        "precedingLabels": [
          "Fill Your Application",
          "Name*"
        ]
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "input.flex.h-10.w-full.rounded-md.border.border-input.bg-background.py-2.ring-offset-background.file:border-0.file:bg-transparent.file:text-sm.file:font-medium.placeholder:text-muted-foreground.focus-visible:outline-none.focus-visible:ring-2.focus-visible:ring-ring.focus-visible:ring-offset-2.disabled:cursor-not-allowed.disabled:opacity-50.flex-1.min-w-[200px].border-none.outline-none.text-sm.placeholder-gray-400.focus:ring-0.px-2:nth-child(1)",
      "id": "",
      "name": "",
      "type": "text",
      "inputType": "text",
      "labels": {
        "placeholder": "Add email address...",
        "precedingLabels": [
          "Fill Your Application",
          "Name*",
          "Email*"
        ]
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "input.form-control:nth-child(2)",
      "id": "",
      "name": "",
      "type": "tel",
      "inputType": "tel",
      "labels": {
        "placeholder": "Enter Contact Number",
        "contextText": "Phone"
      },
      "value": "+91",
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#:r3:-form-item",
      "id": ":r3:-form-item",
      "name": "address",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Address*",
        "placeholder": "Enter Current Address",
        "precedingLabels": [
          "Name*",
          "Email*",
          "Phone*",
          "Phone",
          "Address*"
        ],
        "contextText": "Address*"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#:r4:-form-item",
      "id": ":r4:-form-item",
      "name": "prefLocation",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Preferred Location",
        "placeholder": "Enter Preferences",
        "precedingLabels": [
          "Email*",
          "Phone*",
          "Phone",
          "Address*",
          "Preferred Location"
        ],
        "contextText": "Preferred Location"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#:r5:-form-item",
      "id": ":r5:-form-item",
      "name": "current_ctc",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Current Salary*",
        "placeholder": "Enter Current Salary",
        "precedingLabels": [
          "Phone*",
          "Phone",
          "Address*",
          "Preferred Location",
          "Current Salary*"
        ],
        "contextText": "Current Salary*"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#:r6:-form-item",
      "id": ":r6:-form-item",
      "name": "expected_ctc",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Expected Salary*",
        "placeholder": "Enter Expected Salary",
        "precedingLabels": [
          "Phone",
          "Address*",
          "Preferred Location",
          "Current Salary*",
          "Expected Salary*"
        ],
        "contextText": "Expected Salary*"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#:r7:-form-item",
      "id": ":r7:-form-item",
      "name": "noticePeriod",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Notice Period*",
        "placeholder": "Enter Notice Period",
        "precedingLabels": [
          "Address*",
          "Preferred Location",
          "Current Salary*",
          "Expected Salary*",
          "Notice Period*"
        ],
        "contextText": "Notice Period*"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    }
  ],
  "metadata": {
    "totalFields": 10,
    "requiredFields": 0,
    "emptyFields": 9,
    "formAction": "https://job.10xscale.ai/4846461985313787904",
    "formMethod": "get"
  }
}
'''

parsed_data = json.loads(parsed_data_json)

# Hardcoded personal details
personal_details_json = r'''
{
    "name": "John Doe",
    "firstName": "John",
    "lastName": "Doe",
    "middleName": "Michael",
    "fullName": "John Michael Doe",
    "gender": "Male",
    "age": "20",
    "DOB": {
        "day": "18",
        "month": "11",
        "year": "2004",
        "DateOfBirth": ["18/11/2004", "2004/11/18"]
    },
    "dateOfBirth": ["18/11/2004", "2004/11/18"],
    "birthday": ["18/11/2004", "2004/11/18"],
    "birthDay": "18",
    "birthMonth": "11",
    "birthYear": "2004",
    "email": "john.doe@example.com",
    "emailAddress": "john.doe@example.com",
    "phoneNumber": "9331087882",
    "phone": "9331087882",
    "mobile": "9331087882",
    "cellNumber": "9331087882",
    "homeNumber": "1800 5234 3083",
    "homePhone": "1800 5234 3083",
    "address": {
        "fullAddress": "AXBX BUILDING, 3rd floor Room no 389, PBS colony XYZ road, Nagole, San Francisco 807101, USA",
        "firstLine": "AXBX BUILDING, 3rd floor Room no 389",
        "secondLine": "PBS colony XYZ road, Nagole",
        "street": "XYZ road",
        "streetAddress": "AXBX BUILDING, 3rd floor Room no 389",
        "streetAddressLine2": "PBS colony XYZ road, Nagole",
        "city": "Nagole",
        "state": "San Francisco",
        "postal": "807101",
        "postalCode": "807101",
        "zipCode": "807101",
        "zip": "807101",
        "country": "USA",
        "countryCode": "US"
    },
    "permanentAddress": "AXBX BUILDING, 3rd floor Room no 389, PBS colony XYZ road, Nagole, San Francisco 807101, USA",
    "permanentAddressStreet": "AXBX BUILDING, 3rd floor Room no 389",
    "permanentAddressLine2": "PBS colony XYZ road, Nagole",
    "permanentAddressCity": "Nagole",
    "permanentAddressState": "San Francisco",
    "permanentAddressPostal": "807101",
    "permanentAddressCountry": "USA",
    "schoolAddress": "State University Campus, Building A, Room 101, Education City, California 90210, USA",
    "schoolAddressStreet": "State University Campus, Building A",
    "schoolAddressLine2": "Room 101, Education City",
    "schoolAddressCity": "Education City",
    "schoolAddressState": "California",
    "schoolAddressPostal": "90210",
    "schoolAddressCountry": "USA",
    "occupation": "Software Engineer",
    "jobTitle": "Senior Developer",
    "title": "Senior Developer",
    "company": "Tech Corp",
    "employer": "Tech Corp",
    "industry": "Information Technology",
    "place": "Tech Corp",
    "NoticePeriod": "1 month",
    "PreferredLocation": "San Francisco",
    "education": {
        "level": "Bachelor",
        "degree": "Bachelor of Science",
        "major": "Computer Science",
        "university": "State University",
        "graduationYear": "2026",
        "year": "2026"
    },
    "educationLevel": "Bachelor",
    "educationDegree": "Bachelor of Science",
    "educationMajor": "Computer Science",
    "educationUniversity": "State University",
    "educationYear": "2026",
    "major": "Computer Science",
    "nameOfCollege": "State University",
    "collegeYear": "2026",
    "year": "2026",
    "highSchool": "Lincoln High School - Springfield - 2018-2022",
    "emergencyContactName": "Jane Doe",
    "emergencyContactRelationship": "Sister",
    "emergencyContactPhone": "9876543210",
    "referenceName": "Dr. Robert Smith",
    "referenceRelationship": "Professor",
    "referencePhone": "5551234567",
    "reference2Name": "Sarah Johnson",
    "reference2Relationship": "Former Manager",
    "reference2Phone": "5559876543",
    "nationality": "American",
    "citizenship": "USA",
    "maritalStatus": "Single",
    "preferredName": "Johnny",
    "SSN": "123-45-6789",
    "passport": "AB1234567",
    "driversLicense": "D12345678",
    "website": "www.johndoe.com",
    "linkedin": "linkedin.com/in/johndoe",
    "github": "github.com/johndoe",
    "preferredLanguage": "English",
    "timezone": "PST",
    "WorkAvailability": {
        "availableDays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "availableHours": "9 AM - 5 PM",
        "workAuthorization": "Yes",
        "preferredWorkLocation": "San Fransisco",
        "ExpectedSalary": "100000",
        "NoticePeriod": "1 Month"
    },
    "bankName": "ABC Bank",
    "accountType": "Checking",
    "resume": "Resume.pdf",
    "resumeFile": "Resume.pdf",
    "CV": "Resume.pdf",
    "CVFile": "Resume.pdf"
}
'''

personal_details = json.loads(personal_details_json)


# ✅ Added: CSS selector sanitization utility
def sanitize_selector(selector: str) -> str:
    if not selector:
        return ""
    selector = selector.strip()
    
    # Handle invalid IDs like "#:r3:-form-item" → [id=':r3:-form-item']
    if selector.startswith("#:"):
        return f"[id='{selector[2:]}']"
    
    # Escape internal colons for valid CSS (#foo:bar → #foo\\:bar)
    if selector.startswith("#"):
        selector = re.sub(r':', r'\\\\:', selector)
    
    # Simplify overly complex input selectors (keep first class)
    if selector.startswith(("input.", "div.")):
        parts = selector.split(".")
        if len(parts) > 1:
            return f"{parts[0]}.{parts[1]}"
    
    return selector


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
        
        # Clean response (remove markdown wrappers)
        if result.startswith('```json'):
            result = result.split('```json')[1].split('```')[0].strip()
        elif result.startswith('```'):
            result = result.split('```')[1].split('```')[0].strip()
        
        # Parse JSON to validate
        actions = json.loads(result)

        # ✅ Sanitize all selectors in actions
        if "actions" in actions:
            for act in actions["actions"]:
                act["selector"] = sanitize_selector(act.get("selector", ""))
        if "manual_fields" in actions:
            for field in actions["manual_fields"]:
                field["selector"] = sanitize_selector(field.get("selector", ""))

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