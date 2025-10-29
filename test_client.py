import requests
import json

# Example usage of the API
API_URL = "http://localhost:8000/autofill"

# Sample parsed data (shortened for example)
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
                    }
                },
                {
                    "selector": "#input_13",
                    "id": "input_13",
                    "type": "email",
                    "labels": {
                        "directLabel": "E-mail"
                    }
                }
            ]
        }
    ]
}

# Sample personal details
personal_details = {
    "firstName": "John",
    "lastName": "Doe",
    "email": "john.doe@example.com",
    "phoneNumber": "9331087882",
    "address": {
        "city": "Nagole",
        "state": "San Francisco",
        "postal": "807101",
        "country": "USA"
    }
}

# Make the API call
response = requests.post(
    API_URL,
    json={
        "parsed_data": parsed_data,
        "personal_details": personal_details
    }
)

# Print the response
if response.status_code == 200:
    print("✅ Success!")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)