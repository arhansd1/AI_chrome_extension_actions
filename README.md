# AI_chrome_extension_AUTO_FILLER

This is the AI part of auto form filler in chrome extension .
It gets the parsed HTML data in json format and uses it to build actions to fill the data and sends the actions back to chrome extension to eventually fill form accurately .


## SETUP

1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Set up environment variables:
   - GEMINI_API_KEY: Your Gemini API key
   - GEMINI_MODEL: The Gemini model to use (default: gemini-2.0-flash-lite)
4. Run the API server: python api.py
5. Test the API with the test client: python test_client.py

## API ENDPOINT

# Example usage of the API in Python
import requests

response = requests.post(
    "http://localhost:8000/autofill",
    json={
        "parsed_data": your_form_data,
        "personal_details": user_details
    }
)
actions = response.json()


# Example usage of the API in JavaScript
const response = await fetch('http://localhost:8000/autofill', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    parsed_data: formData,
    personal_details: userDetails
  })
});
const actions = await response.json();