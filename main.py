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

# Hardcoded parsed data (DUMMY - will be replaced at runtime)
parsed_data_json = r'''
{
  "url": "https://form.jotform.com/252131352654450",
  "title": "Sample Job Application Form",
  "timestamp": "2025-10-27T05:31:31.416Z",
  "sections": [
    {
      "heading": "Personal Information",
      "fields": [
        {
          "selector": "#first_4",
          "id": "first_4",
          "name": "q4_name[first]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Name",
            "placeholder": "",
            "contextText": "First Name"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#last_4",
          "id": "last_4",
          "name": "q4_name[last]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Last Name",
            "placeholder": "",
            "contextText": "Last Name"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_5_full",
          "id": "input_5_full",
          "name": "q5_phoneNumber[full]",
          "type": "tel",
          "inputType": "tel",
          "labels": {
            "directLabel": "Phone Number",
            "placeholder": "(000) 000-0000"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_6",
          "id": "input_6",
          "name": "q6_email",
          "type": "email",
          "inputType": "email",
          "labels": {
            "directLabel": "Email",
            "placeholder": "",
            "contextText": "example@example.com"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_7_addr_line1",
          "id": "input_7_addr_line1",
          "name": "q7_address[addr_line1]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Address",
            "placeholder": "",
            "contextText": "Street Address"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {
            "maxLength": 100
          }
        },
        {
          "selector": "#input_7_addr_line2",
          "id": "input_7_addr_line2",
          "name": "q7_address[addr_line2]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Street Address Line 2",
            "placeholder": "",
            "contextText": "Street Address Line 2"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {
            "maxLength": 100
          }
        },
        {
          "selector": "#input_7_city",
          "id": "input_7_city",
          "name": "q7_address[city]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "City",
            "placeholder": "",
            "contextText": "City"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {
            "maxLength": 60
          }
        },
        {
          "selector": "#input_7_state",
          "id": "input_7_state",
          "name": "q7_address[state]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "State / Province",
            "placeholder": "",
            "contextText": "State / Province"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {
            "maxLength": 60
          }
        },
        {
          "selector": "#input_7_postal",
          "id": "input_7_postal",
          "name": "q7_address[postal]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Postal / Zip Code",
            "placeholder": "",
            "contextText": "Postal / Zip Code"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {
            "maxLength": 20
          }
        },
        {
          "selector": "#input_10",
          "id": "input_10",
          "name": "q10_whatIs",
          "type": "select-one",
          "labels": {
            "directLabel": "What is the best time to contact you?",
            "ariaLabel": "What is the best time to contact you?"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "options": {
            "unselected": [
              [
                "Morning",
                "Morning"
              ],
              [
                "Lunch Time",
                "Lunch Time"
              ],
              [
                "Evening",
                "Evening"
              ],
              [
                "Afternoon",
                "Afternoon"
              ],
              [
                "Doesn't Matter",
                "Doesn't Matter"
              ]
            ],
            "selected": [
              [
                "",
                "Please Select"
              ]
            ]
          },
          "validation": {}
        },
        {
          "selector": "#input_8_0",
          "id": "input_8_0",
          "name": "q8_areYou8",
          "type": "radio",
          "inputType": "radio",
          "labels": {
            "directLabel": "Yes",
            "groupLabel": "Are you currently legally entitled to work in the country where the job is based?",
            "placeholder": "",
            "contextText": "Yes"
          },
          "value": false,
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_8_1",
          "id": "input_8_1",
          "name": "q8_areYou8",
          "type": "radio",
          "inputType": "radio",
          "labels": {
            "directLabel": "No",
            "groupLabel": "Are you currently legally entitled to work in the country where the job is based?",
            "placeholder": "",
            "contextText": "No"
          },
          "value": false,
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_14",
          "id": "input_14",
          "name": "q14_ifApplicable14",
          "type": "textarea",
          "labels": {
            "directLabel": "If applicable, please detail any restrictions:",
            "placeholder": ""
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_9_0",
          "id": "input_9_0",
          "name": "q9_ifSelected",
          "type": "radio",
          "inputType": "radio",
          "labels": {
            "directLabel": "Yes",
            "groupLabel": "If selected for employment are you willing to submit a background check?",
            "placeholder": "",
            "contextText": "Yes"
          },
          "value": false,
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_9_1",
          "id": "input_9_1",
          "name": "q9_ifSelected",
          "type": "radio",
          "inputType": "radio",
          "labels": {
            "directLabel": "No",
            "groupLabel": "If selected for employment are you willing to submit a background check?",
            "placeholder": "",
            "contextText": "No"
          },
          "value": false,
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
      "selector": "#first_4",
      "id": "first_4",
      "name": "q4_name[first]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Name",
        "placeholder": "",
        "contextText": "First Name"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#last_4",
      "id": "last_4",
      "name": "q4_name[last]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Last Name",
        "placeholder": "",
        "contextText": "Last Name"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_5_full",
      "id": "input_5_full",
      "name": "q5_phoneNumber[full]",
      "type": "tel",
      "inputType": "tel",
      "labels": {
        "directLabel": "Phone Number",
        "placeholder": "(000) 000-0000"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_6",
      "id": "input_6",
      "name": "q6_email",
      "type": "email",
      "inputType": "email",
      "labels": {
        "directLabel": "Email",
        "placeholder": "",
        "contextText": "example@example.com"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_7_addr_line1",
      "id": "input_7_addr_line1",
      "name": "q7_address[addr_line1]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Address",
        "placeholder": "",
        "contextText": "Street Address"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {
        "maxLength": 100
      }
    },
    {
      "selector": "#input_7_addr_line2",
      "id": "input_7_addr_line2",
      "name": "q7_address[addr_line2]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Street Address Line 2",
        "placeholder": "",
        "contextText": "Street Address Line 2"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {
        "maxLength": 100
      }
    },
    {
      "selector": "#input_7_city",
      "id": "input_7_city",
      "name": "q7_address[city]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "City",
        "placeholder": "",
        "contextText": "City"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {
        "maxLength": 60
      }
    },
    {
      "selector": "#input_7_state",
      "id": "input_7_state",
      "name": "q7_address[state]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "State / Province",
        "placeholder": "",
        "contextText": "State / Province"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {
        "maxLength": 60
      }
    },
    {
      "selector": "#input_7_postal",
      "id": "input_7_postal",
      "name": "q7_address[postal]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Postal / Zip Code",
        "placeholder": "",
        "contextText": "Postal / Zip Code"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {
        "maxLength": 20
      }
    },
    {
      "selector": "#input_10",
      "id": "input_10",
      "name": "q10_whatIs",
      "type": "select-one",
      "labels": {
        "directLabel": "What is the best time to contact you?",
        "ariaLabel": "What is the best time to contact you?"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "options": {
        "unselected": [
          [
            "Morning",
            "Morning"
          ],
          [
            "Lunch Time",
            "Lunch Time"
          ],
          [
            "Evening",
            "Evening"
          ],
          [
            "Afternoon",
            "Afternoon"
          ],
          [
            "Doesn't Matter",
            "Doesn't Matter"
          ]
        ],
        "selected": [
          [
            "",
            "Please Select"
          ]
        ]
      },
      "validation": {}
    },
    {
      "selector": "#input_8_0",
      "id": "input_8_0",
      "name": "q8_areYou8",
      "type": "radio",
      "inputType": "radio",
      "labels": {
        "directLabel": "Yes",
        "groupLabel": "Are you currently legally entitled to work in the country where the job is based?",
        "placeholder": "",
        "contextText": "Yes"
      },
      "value": false,
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_8_1",
      "id": "input_8_1",
      "name": "q8_areYou8",
      "type": "radio",
      "inputType": "radio",
      "labels": {
        "directLabel": "No",
        "groupLabel": "Are you currently legally entitled to work in the country where the job is based?",
        "placeholder": "",
        "contextText": "No"
      },
      "value": false,
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_14",
      "id": "input_14",
      "name": "q14_ifApplicable14",
      "type": "textarea",
      "labels": {
        "directLabel": "If applicable, please detail any restrictions:",
        "placeholder": ""
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_9_0",
      "id": "input_9_0",
      "name": "q9_ifSelected",
      "type": "radio",
      "inputType": "radio",
      "labels": {
        "directLabel": "Yes",
        "groupLabel": "If selected for employment are you willing to submit a background check?",
        "placeholder": "",
        "contextText": "Yes"
      },
      "value": false,
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_9_1",
      "id": "input_9_1",
      "name": "q9_ifSelected",
      "type": "radio",
      "inputType": "radio",
      "labels": {
        "directLabel": "No",
        "groupLabel": "If selected for employment are you willing to submit a background check?",
        "placeholder": "",
        "contextText": "No"
      },
      "value": false,
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    }
  ],
  "metadata": {
    "totalFields": 15,
    "requiredFields": 0,
    "emptyFields": 15,
    "formAction": "https://submit.jotform.com/submit/252131352654450",
    "formMethod": "post"
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
    "skills": ["JavaScript", "Python", "React", "Node.js"],
    "languages": ["English", "Spanish"],
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


def sanitize_selector(selector: str) -> str:
    """Sanitize CSS selectors for edge cases"""
    if not selector:
        return ""
    selector = selector.strip()
    
    # Handle invalid IDs like "#:r3:-form-item" → [id=':r3:-form-item']
    if selector.startswith("#:"):
        return f"[id='{selector[1:]}']"
    
    # Don't escape colons - leave selectors as-is from parsed data
    # The content script should handle them properly
    
    return selector


def call_llm(parsed_data, personal_details):
    """Call Gemini to generate autofill actions with all action types"""
    
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
        print(f"❌ JSON Parse Error: {e}")
        print(f"Raw response: {result[:500]}")
        return None
    except Exception as e:
        print(f"❌ Error calling LLM: {e}")
        return None


def validate_actions(actions):
    """Validate generated actions for completeness"""
    if not actions or "actions" not in actions:
        print("⚠️  No actions generated")
        return False
    
    action_types = {}
    for action in actions["actions"]:
        action_type = action.get("action")
        action_types[action_type] = action_types.get(action_type, 0) + 1
    
    print("\n📊 Action Type Distribution:")
    for action_type, count in sorted(action_types.items()):
        print(f"   {action_type}: {count}")
    
    # Check for essential action types
    essential_types = ["fill", "select"]
    missing_types = [t for t in essential_types if t not in action_types]
    
    if missing_types:
        print(f"\n⚠️  Warning: Missing essential action types: {missing_types}")
    
    return True


def main():
    print("=" * 60)
    print("AI FORM AUTOFILL - ENHANCED WITH ALL ACTION TYPES")
    print("=" * 60)
    
    print("\n📋 Parsed Form Data:")
    print(f"   Fields: {len(parsed_data.get('allFields', []))}")
    print(f"   Sections: {len(parsed_data.get('sections', []))}")
    
    print("\n👤 Personal Details:")
    print(f"   Name: {personal_details.get('fullName')}")
    print(f"   Email: {personal_details.get('email')}")
    print(f"   Phone: {personal_details.get('phoneNumber')}")
    print(f"   Available Keys: {len(personal_details.keys())}")
    
    print("\n🤖 Calling Gemini LLM...")
    print("   Generating actions for:")
    print("   ✓ Text fields (fill)")
    print("   ✓ Dropdowns (select, select_multiple)")
    print("   ✓ Radio buttons (radio_select)")
    print("   ✓ Checkboxes (check, uncheck, check_multiple)")
    print("   ✓ Date fields (date_fill)")
    print("   ✓ File uploads (upload_file)")
    print("   ✓ Autocomplete (autocomplete)")
    
    actions = call_llm(parsed_data, personal_details)
    
    if actions:
        print("\n✅ Generated Actions Successfully!")
        
        # Validate actions
        validate_actions(actions)
        
        print(f"\n📝 Total Actions: {len(actions.get('actions', []))}")
        if "manual_fields" in actions and actions["manual_fields"]:
            print(f"⚠️  Manual Fields: {len(actions['manual_fields'])}")
        
        # Display sample actions
        print("\n📋 Sample Actions (first 5):")
        for i, action in enumerate(actions.get("actions", [])[:5], 1):
            print(f"\n   {i}. {action.get('action').upper()}")
            print(f"      Selector: {action.get('selector')}")
            print(f"      Value: {action.get('value')}")
            print(f"      Confidence: {action.get('confidence')}")
            print(f"      Reasoning: {action.get('reasoning')}")
        
        # Save to file
        output_file = 'output_actions.json'
        with open(output_file, 'w') as f:
            json.dump(actions, f, indent=2)
        print(f"\n💾 Actions saved to {output_file}")
        
        # Display full JSON for debugging
        print("\n" + "=" * 60)
        print("COMPLETE JSON OUTPUT:")
        print("=" * 60)
        print(json.dumps(actions, indent=2))
        
    else:
        print("\n❌ Failed to generate actions")
        print("   Check your GEMINI_API_KEY and network connection")

if __name__ == '__main__':
    main()


