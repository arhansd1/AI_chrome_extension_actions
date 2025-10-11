# main.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash-exp')

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
      "optional": false
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
- fill: text inputs
- check: checkboxes/radio (value: true)
- uncheck: checkboxes (value: false)
- select: dropdowns (value: option text)

Return ONLY valid JSON, no explanations."""

@app.route('/autofill', methods=['POST'])
def autofill():
    try:
        data = request.json
        parsed_data = data.get('parsed_data', {})
        personal_details = data.get('personal_details', {})
        
        # Construct prompt
        prompt = f"""
{SYSTEM_PROMPT}

FORM DATA:
{parsed_data}

PERSONAL DETAILS:
{personal_details}

Generate autofill actions:"""

        # Call Gemini
        response = model.generate_content(prompt)
        result = response.text
        
        # Clean response (remove markdown if present)
        if result.startswith('```json'):
            result = result.split('```json')[1].split('```')[0].strip()
        elif result.startswith('```'):
            result = result.split('```')[1].split('```')[0].strip()
        
        return jsonify({
            "success": True,
            "data": result
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)