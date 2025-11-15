import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash-lite')

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
  "url": "https://form.jotform.com/252470591559465",
  "title": "Online Internship Application Form",
  "timestamp": "2025-10-29T06:05:10.757Z",
  "sections": [
    {
      "heading": "Personal Information",
      "fields": [
        {
          "selector": "#first_12",
          "id": "first_12",
          "name": "q12_fullName12[first]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Full Name",
            "placeholder": "",
            "contextText": "First Name"
          },
          "value": "John",
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#last_12",
          "id": "last_12",
          "name": "q12_fullName12[last]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Last Name",
            "placeholder": "",
            "contextText": "Last Name"
          },
          "value": "Doe",
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_54",
          "id": "input_54",
          "name": "q54_preferredName54",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Preferred Name",
            "placeholder": " "
          },
          "value": "Johnny",
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_20_0",
          "id": "input_20_0",
          "name": "q20_gender20[]",
          "type": "checkbox",
          "inputType": "checkbox",
          "labels": {
            "directLabel": "Male",
            "groupLabel": "Gender",
            "placeholder": "",
            "contextText": "Male"
          },
          "value": true,
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_20_1",
          "id": "input_20_1",
          "name": "q20_gender20[]",
          "type": "checkbox",
          "inputType": "checkbox",
          "labels": {
            "directLabel": "Female",
            "groupLabel": "Gender",
            "placeholder": "",
            "contextText": "Female"
          },
          "value": false,
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_19_month",
          "id": "input_19_month",
          "name": "q19_birthDate19[month]",
          "type": "select-one",
          "labels": {
            "directLabel": "Month",
            "contextText": "Month"
          },
          "value": "11",
          "isEmpty": false,
          "isRequired": false,
          "options": {
            "unselected": [
              [
                "",
                "Please select a month"
              ],
              [
                "1",
                "January"
              ],
              [
                "2",
                "February"
              ],
              [
                "3",
                "March"
              ],
              [
                "4",
                "April"
              ],
              [
                "5",
                "May"
              ],
              [
                "6",
                "June"
              ],
              [
                "7",
                "July"
              ],
              [
                "8",
                "August"
              ],
              [
                "9",
                "September"
              ],
              [
                "10",
                "October"
              ],
              [
                "12",
                "December"
              ]
            ],
            "selected": [
              [
                "11",
                "November"
              ]
            ]
          },
          "validation": {}
        },
        {
          "selector": "#input_19_day",
          "id": "input_19_day",
          "name": "q19_birthDate19[day]",
          "type": "select-one",
          "labels": {
            "directLabel": "Day",
            "contextText": "Day"
          },
          "value": "18",
          "isEmpty": false,
          "isRequired": false,
          "options": {
            "unselected": [
              [
                "",
                "Please select a day"
              ],
              [
                "1",
                "1"
              ],
              [
                "2",
                "2"
              ],
              [
                "3",
                "3"
              ],
              [
                "4",
                "4"
              ],
              [
                "5",
                "5"
              ],
              [
                "6",
                "6"
              ],
              [
                "7",
                "7"
              ],
              [
                "8",
                "8"
              ],
              [
                "9",
                "9"
              ],
              [
                "10",
                "10"
              ],
              [
                "11",
                "11"
              ],
              [
                "12",
                "12"
              ],
              [
                "13",
                "13"
              ],
              [
                "14",
                "14"
              ],
              [
                "15",
                "15"
              ],
              [
                "16",
                "16"
              ],
              [
                "17",
                "17"
              ],
              [
                "19",
                "19"
              ],
              [
                "20",
                "20"
              ],
              [
                "21",
                "21"
              ],
              [
                "22",
                "22"
              ],
              [
                "23",
                "23"
              ],
              [
                "24",
                "24"
              ],
              [
                "25",
                "25"
              ],
              [
                "26",
                "26"
              ],
              [
                "27",
                "27"
              ],
              [
                "28",
                "28"
              ],
              [
                "29",
                "29"
              ],
              [
                "30",
                "30"
              ],
              [
                "31",
                "31"
              ]
            ],
            "selected": [
              [
                "18",
                "18"
              ]
            ]
          },
          "validation": {}
        },
        {
          "selector": "#input_19_year",
          "id": "input_19_year",
          "name": "q19_birthDate19[year]",
          "type": "select-one",
          "labels": {
            "directLabel": "Year",
            "contextText": "Year"
          },
          "value": "2004",
          "isEmpty": false,
          "isRequired": false,
          "options": {
            "unselected": [
              [
                "",
                "Please select a year"
              ],
              [
                "2025",
                "2025"
              ],
              [
                "2024",
                "2024"
              ],
              [
                "2023",
                "2023"
              ],
              [
                "2022",
                "2022"
              ],
              [
                "2021",
                "2021"
              ],
              [
                "2020",
                "2020"
              ],
              [
                "2019",
                "2019"
              ],
              [
                "2018",
                "2018"
              ],
              [
                "2017",
                "2017"
              ],
              [
                "2016",
                "2016"
              ],
              [
                "2015",
                "2015"
              ],
              [
                "2014",
                "2014"
              ],
              [
                "2013",
                "2013"
              ],
              [
                "2012",
                "2012"
              ],
              [
                "2011",
                "2011"
              ],
              [
                "2010",
                "2010"
              ],
              [
                "2009",
                "2009"
              ],
              [
                "2008",
                "2008"
              ],
              [
                "2007",
                "2007"
              ],
              [
                "2006",
                "2006"
              ],
              [
                "2005",
                "2005"
              ],
              [
                "2003",
                "2003"
              ],
              [
                "2002",
                "2002"
              ],
              [
                "2001",
                "2001"
              ],
              [
                "2000",
                "2000"
              ],
              [
                "1999",
                "1999"
              ],
              [
                "1998",
                "1998"
              ],
              [
                "1997",
                "1997"
              ],
              [
                "1996",
                "1996"
              ],
              [
                "1995",
                "1995"
              ],
              [
                "1994",
                "1994"
              ],
              [
                "1993",
                "1993"
              ],
              [
                "1992",
                "1992"
              ],
              [
                "1991",
                "1991"
              ],
              [
                "1990",
                "1990"
              ],
              [
                "1989",
                "1989"
              ],
              [
                "1988",
                "1988"
              ],
              [
                "1987",
                "1987"
              ],
              [
                "1986",
                "1986"
              ],
              [
                "1985",
                "1985"
              ],
              [
                "1984",
                "1984"
              ],
              [
                "1983",
                "1983"
              ],
              [
                "1982",
                "1982"
              ],
              [
                "1981",
                "1981"
              ],
              [
                "1980",
                "1980"
              ],
              [
                "1979",
                "1979"
              ],
              [
                "1978",
                "1978"
              ],
              [
                "1977",
                "1977"
              ],
              [
                "1976",
                "1976"
              ],
              [
                "1975",
                "1975"
              ],
              [
                "1974",
                "1974"
              ],
              [
                "1973",
                "1973"
              ],
              [
                "1972",
                "1972"
              ],
              [
                "1971",
                "1971"
              ],
              [
                "1970",
                "1970"
              ],
              [
                "1969",
                "1969"
              ],
              [
                "1968",
                "1968"
              ],
              [
                "1967",
                "1967"
              ],
              [
                "1966",
                "1966"
              ],
              [
                "1965",
                "1965"
              ],
              [
                "1964",
                "1964"
              ],
              [
                "1963",
                "1963"
              ],
              [
                "1962",
                "1962"
              ],
              [
                "1961",
                "1961"
              ],
              [
                "1960",
                "1960"
              ],
              [
                "1959",
                "1959"
              ],
              [
                "1958",
                "1958"
              ],
              [
                "1957",
                "1957"
              ],
              [
                "1956",
                "1956"
              ],
              [
                "1955",
                "1955"
              ],
              [
                "1954",
                "1954"
              ],
              [
                "1953",
                "1953"
              ],
              [
                "1952",
                "1952"
              ],
              [
                "1951",
                "1951"
              ],
              [
                "1950",
                "1950"
              ],
              [
                "1949",
                "1949"
              ],
              [
                "1948",
                "1948"
              ],
              [
                "1947",
                "1947"
              ],
              [
                "1946",
                "1946"
              ],
              [
                "1945",
                "1945"
              ],
              [
                "1944",
                "1944"
              ],
              [
                "1943",
                "1943"
              ],
              [
                "1942",
                "1942"
              ],
              [
                "1941",
                "1941"
              ],
              [
                "1940",
                "1940"
              ],
              [
                "1939",
                "1939"
              ],
              [
                "1938",
                "1938"
              ],
              [
                "1937",
                "1937"
              ],
              [
                "1936",
                "1936"
              ],
              [
                "1935",
                "1935"
              ],
              [
                "1934",
                "1934"
              ],
              [
                "1933",
                "1933"
              ],
              [
                "1932",
                "1932"
              ],
              [
                "1931",
                "1931"
              ],
              [
                "1930",
                "1930"
              ],
              [
                "1929",
                "1929"
              ],
              [
                "1928",
                "1928"
              ],
              [
                "1927",
                "1927"
              ],
              [
                "1926",
                "1926"
              ],
              [
                "1925",
                "1925"
              ],
              [
                "1924",
                "1924"
              ],
              [
                "1923",
                "1923"
              ],
              [
                "1922",
                "1922"
              ],
              [
                "1921",
                "1921"
              ],
              [
                "1920",
                "1920"
              ]
            ],
            "selected": [
              [
                "2004",
                "2004"
              ]
            ]
          },
          "validation": {}
        },
        {
          "selector": "#input_55",
          "id": "input_55",
          "name": "q55_age",
          "type": "number",
          "inputType": "number",
          "labels": {
            "directLabel": "Age",
            "placeholder": ""
          },
          "value": "20",
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_13",
          "id": "input_13",
          "name": "q13_email13",
          "type": "email",
          "inputType": "email",
          "labels": {
            "directLabel": "E-mail",
            "placeholder": "ex: myname@example.com",
            "contextText": "example@example.com"
          },
          "value": "john.doe@example.com",
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_17_full",
          "id": "input_17_full",
          "name": "q17_homeNumber[full]",
          "type": "tel",
          "inputType": "tel",
          "labels": {
            "directLabel": "Home Number",
            "placeholder": ""
          },
          "value": "(180) 052-3430",
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_18_full",
          "id": "input_18_full",
          "name": "q18_cellNumber[full]",
          "type": "tel",
          "inputType": "tel",
          "labels": {
            "directLabel": "Cell Number",
            "placeholder": ""
          },
          "value": "(933) 108-7882",
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_15_addr_line1",
          "id": "input_15_addr_line1",
          "name": "q15_permanentAddress[addr_line1]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Permanent Address",
            "placeholder": "",
            "contextText": "Street Address"
          },
          "value": "AXBX BUILDING, 3rd floor Room no 389",
          "isEmpty": false,
          "isRequired": false,
          "validation": {
            "maxLength": 100
          }
        },
        {
          "selector": "#input_15_addr_line2",
          "id": "input_15_addr_line2",
          "name": "q15_permanentAddress[addr_line2]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Street Address Line 2",
            "placeholder": "",
            "contextText": "Street Address Line 2"
          },
          "value": "PBS colony XYZ road, Nagole",
          "isEmpty": false,
          "isRequired": false,
          "validation": {
            "maxLength": 100
          }
        },
        {
          "selector": "#input_15_city",
          "id": "input_15_city",
          "name": "q15_permanentAddress[city]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "City",
            "placeholder": "",
            "contextText": "City"
          },
          "value": "Nagole",
          "isEmpty": false,
          "isRequired": false,
          "validation": {
            "maxLength": 60
          }
        },
        {
          "selector": "#input_15_state",
          "id": "input_15_state",
          "name": "q15_permanentAddress[state]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "State / Province",
            "placeholder": "",
            "contextText": "State / Province"
          },
          "value": "San Francisco",
          "isEmpty": false,
          "isRequired": false,
          "validation": {
            "maxLength": 60
          }
        },
        {
          "selector": "#input_15_postal",
          "id": "input_15_postal",
          "name": "q15_permanentAddress[postal]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Postal / Zip Code",
            "placeholder": "",
            "contextText": "Postal / Zip Code"
          },
          "value": "807101",
          "isEmpty": false,
          "isRequired": false,
          "validation": {
            "maxLength": 20
          }
        },
        {
          "selector": "#input_15_country",
          "id": "input_15_country",
          "name": "q15_permanentAddress[country]",
          "type": "select-one",
          "labels": {
            "directLabel": "Country",
            "contextText": "Country"
          },
          "value": "United States",
          "isEmpty": false,
          "isRequired": true,
          "options": {
            "unselected": [
              [
                "",
                "Please Select"
              ],
              [
                "Afghanistan",
                "Afghanistan"
              ],
              [
                "Albania",
                "Albania"
              ],
              [
                "Algeria",
                "Algeria"
              ],
              [
                "American Samoa",
                "American Samoa"
              ],
              [
                "Andorra",
                "Andorra"
              ],
              [
                "Angola",
                "Angola"
              ],
              [
                "Anguilla",
                "Anguilla"
              ],
              [
                "Antigua and Barbuda",
                "Antigua and Barbuda"
              ],
              [
                "Argentina",
                "Argentina"
              ],
              [
                "Armenia",
                "Armenia"
              ],
              [
                "Aruba",
                "Aruba"
              ],
              [
                "Australia",
                "Australia"
              ],
              [
                "Austria",
                "Austria"
              ],
              [
                "Azerbaijan",
                "Azerbaijan"
              ],
              [
                "The Bahamas",
                "The Bahamas"
              ],
              [
                "Bahrain",
                "Bahrain"
              ],
              [
                "Bangladesh",
                "Bangladesh"
              ],
              [
                "Barbados",
                "Barbados"
              ],
              [
                "Belarus",
                "Belarus"
              ],
              [
                "Belgium",
                "Belgium"
              ],
              [
                "Belize",
                "Belize"
              ],
              [
                "Benin",
                "Benin"
              ],
              [
                "Bermuda",
                "Bermuda"
              ],
              [
                "Bhutan",
                "Bhutan"
              ],
              [
                "Bolivia",
                "Bolivia"
              ],
              [
                "Bosnia and Herzegovina",
                "Bosnia and Herzegovina"
              ],
              [
                "Botswana",
                "Botswana"
              ],
              [
                "Brazil",
                "Brazil"
              ],
              [
                "Brunei",
                "Brunei"
              ],
              [
                "Bulgaria",
                "Bulgaria"
              ],
              [
                "Burkina Faso",
                "Burkina Faso"
              ],
              [
                "Burundi",
                "Burundi"
              ],
              [
                "Cambodia",
                "Cambodia"
              ],
              [
                "Cameroon",
                "Cameroon"
              ],
              [
                "Canada",
                "Canada"
              ],
              [
                "Cape Verde",
                "Cape Verde"
              ],
              [
                "Cayman Islands",
                "Cayman Islands"
              ],
              [
                "Central African Republic",
                "Central African Republic"
              ],
              [
                "Chad",
                "Chad"
              ],
              [
                "Chile",
                "Chile"
              ],
              [
                "China",
                "China"
              ],
              [
                "Christmas Island",
                "Christmas Island"
              ],
              [
                "Cocos (Keeling) Islands",
                "Cocos (Keeling) Islands"
              ],
              [
                "Colombia",
                "Colombia"
              ],
              [
                "Comoros",
                "Comoros"
              ],
              [
                "Congo",
                "Congo"
              ],
              [
                "Cook Islands",
                "Cook Islands"
              ],
              [
                "Costa Rica",
                "Costa Rica"
              ],
              [
                "Cote d'Ivoire",
                "Cote d'Ivoire"
              ],
              [
                "Croatia",
                "Croatia"
              ],
              [
                "Cuba",
                "Cuba"
              ],
              [
                "Curaçao",
                "Curaçao"
              ],
              [
                "Cyprus",
                "Cyprus"
              ],
              [
                "Czech Republic",
                "Czech Republic"
              ],
              [
                "Democratic Republic of the Congo",
                "Democratic Republic of the Congo"
              ],
              [
                "Denmark",
                "Denmark"
              ],
              [
                "Djibouti",
                "Djibouti"
              ],
              [
                "Dominica",
                "Dominica"
              ],
              [
                "Dominican Republic",
                "Dominican Republic"
              ],
              [
                "Ecuador",
                "Ecuador"
              ],
              [
                "Egypt",
                "Egypt"
              ],
              [
                "El Salvador",
                "El Salvador"
              ],
              [
                "Equatorial Guinea",
                "Equatorial Guinea"
              ],
              [
                "Eritrea",
                "Eritrea"
              ],
              [
                "Estonia",
                "Estonia"
              ],
              [
                "Ethiopia",
                "Ethiopia"
              ],
              [
                "Falkland Islands",
                "Falkland Islands"
              ],
              [
                "Faroe Islands",
                "Faroe Islands"
              ],
              [
                "Fiji",
                "Fiji"
              ],
              [
                "Finland",
                "Finland"
              ],
              [
                "France",
                "France"
              ],
              [
                "French Polynesia",
                "French Polynesia"
              ],
              [
                "Gabon",
                "Gabon"
              ],
              [
                "The Gambia",
                "The Gambia"
              ],
              [
                "Georgia",
                "Georgia"
              ],
              [
                "Germany",
                "Germany"
              ],
              [
                "Ghana",
                "Ghana"
              ],
              [
                "Gibraltar",
                "Gibraltar"
              ],
              [
                "Greece",
                "Greece"
              ],
              [
                "Greenland",
                "Greenland"
              ],
              [
                "Grenada",
                "Grenada"
              ],
              [
                "Guadeloupe",
                "Guadeloupe"
              ],
              [
                "Guam",
                "Guam"
              ],
              [
                "Guatemala",
                "Guatemala"
              ],
              [
                "Guernsey",
                "Guernsey"
              ],
              [
                "Guinea",
                "Guinea"
              ],
              [
                "Guinea-Bissau",
                "Guinea-Bissau"
              ],
              [
                "Guyana",
                "Guyana"
              ],
              [
                "Haiti",
                "Haiti"
              ],
              [
                "Honduras",
                "Honduras"
              ],
              [
                "Hong Kong",
                "Hong Kong"
              ],
              [
                "Hungary",
                "Hungary"
              ],
              [
                "Iceland",
                "Iceland"
              ],
              [
                "India",
                "India"
              ],
              [
                "Indonesia",
                "Indonesia"
              ],
              [
                "Iran",
                "Iran"
              ],
              [
                "Iraq",
                "Iraq"
              ],
              [
                "Ireland",
                "Ireland"
              ],
              [
                "Israel",
                "Israel"
              ],
              [
                "Italy",
                "Italy"
              ],
              [
                "Jamaica",
                "Jamaica"
              ],
              [
                "Japan",
                "Japan"
              ],
              [
                "Jersey",
                "Jersey"
              ],
              [
                "Jordan",
                "Jordan"
              ],
              [
                "Kazakhstan",
                "Kazakhstan"
              ],
              [
                "Kenya",
                "Kenya"
              ],
              [
                "Kiribati",
                "Kiribati"
              ],
              [
                "North Korea",
                "North Korea"
              ],
              [
                "South Korea",
                "South Korea"
              ],
              [
                "Kosovo",
                "Kosovo"
              ],
              [
                "Kuwait",
                "Kuwait"
              ],
              [
                "Kyrgyzstan",
                "Kyrgyzstan"
              ],
              [
                "Laos",
                "Laos"
              ],
              [
                "Latvia",
                "Latvia"
              ],
              [
                "Lebanon",
                "Lebanon"
              ],
              [
                "Lesotho",
                "Lesotho"
              ],
              [
                "Liberia",
                "Liberia"
              ],
              [
                "Libya",
                "Libya"
              ],
              [
                "Liechtenstein",
                "Liechtenstein"
              ],
              [
                "Lithuania",
                "Lithuania"
              ],
              [
                "Luxembourg",
                "Luxembourg"
              ],
              [
                "Macau",
                "Macau"
              ],
              [
                "Macedonia",
                "Macedonia"
              ],
              [
                "Madagascar",
                "Madagascar"
              ],
              [
                "Malawi",
                "Malawi"
              ],
              [
                "Malaysia",
                "Malaysia"
              ],
              [
                "Maldives",
                "Maldives"
              ],
              [
                "Mali",
                "Mali"
              ],
              [
                "Malta",
                "Malta"
              ],
              [
                "Marshall Islands",
                "Marshall Islands"
              ],
              [
                "Martinique",
                "Martinique"
              ],
              [
                "Mauritania",
                "Mauritania"
              ],
              [
                "Mauritius",
                "Mauritius"
              ],
              [
                "Mayotte",
                "Mayotte"
              ],
              [
                "Mexico",
                "Mexico"
              ],
              [
                "Micronesia",
                "Micronesia"
              ],
              [
                "Moldova",
                "Moldova"
              ],
              [
                "Monaco",
                "Monaco"
              ],
              [
                "Mongolia",
                "Mongolia"
              ],
              [
                "Montenegro",
                "Montenegro"
              ],
              [
                "Montserrat",
                "Montserrat"
              ],
              [
                "Morocco",
                "Morocco"
              ],
              [
                "Mozambique",
                "Mozambique"
              ],
              [
                "Myanmar",
                "Myanmar"
              ],
              [
                "Nagorno-Karabakh",
                "Nagorno-Karabakh"
              ],
              [
                "Namibia",
                "Namibia"
              ],
              [
                "Nauru",
                "Nauru"
              ],
              [
                "Nepal",
                "Nepal"
              ],
              [
                "Netherlands",
                "Netherlands"
              ],
              [
                "Netherlands Antilles",
                "Netherlands Antilles"
              ],
              [
                "New Caledonia",
                "New Caledonia"
              ],
              [
                "New Zealand",
                "New Zealand"
              ],
              [
                "Nicaragua",
                "Nicaragua"
              ],
              [
                "Niger",
                "Niger"
              ],
              [
                "Nigeria",
                "Nigeria"
              ],
              [
                "Niue",
                "Niue"
              ],
              [
                "Norfolk Island",
                "Norfolk Island"
              ],
              [
                "Turkish Republic of Northern Cyprus",
                "Turkish Republic of Northern Cyprus"
              ],
              [
                "Northern Mariana",
                "Northern Mariana"
              ],
              [
                "Norway",
                "Norway"
              ],
              [
                "Oman",
                "Oman"
              ],
              [
                "Pakistan",
                "Pakistan"
              ],
              [
                "Palau",
                "Palau"
              ],
              [
                "Palestine",
                "Palestine"
              ],
              [
                "Panama",
                "Panama"
              ],
              [
                "Papua New Guinea",
                "Papua New Guinea"
              ],
              [
                "Paraguay",
                "Paraguay"
              ],
              [
                "Peru",
                "Peru"
              ],
              [
                "Philippines",
                "Philippines"
              ],
              [
                "Pitcairn Islands",
                "Pitcairn Islands"
              ],
              [
                "Poland",
                "Poland"
              ],
              [
                "Portugal",
                "Portugal"
              ],
              [
                "Puerto Rico",
                "Puerto Rico"
              ],
              [
                "Qatar",
                "Qatar"
              ],
              [
                "Republic of the Congo",
                "Republic of the Congo"
              ],
              [
                "Romania",
                "Romania"
              ],
              [
                "Russia",
                "Russia"
              ],
              [
                "Rwanda",
                "Rwanda"
              ],
              [
                "Saint Barthelemy",
                "Saint Barthelemy"
              ],
              [
                "Saint Helena",
                "Saint Helena"
              ],
              [
                "Saint Kitts and Nevis",
                "Saint Kitts and Nevis"
              ],
              [
                "Saint Lucia",
                "Saint Lucia"
              ],
              [
                "Saint Martin",
                "Saint Martin"
              ],
              [
                "Saint Pierre and Miquelon",
                "Saint Pierre and Miquelon"
              ],
              [
                "Saint Vincent and the Grenadines",
                "Saint Vincent and the Grenadines"
              ],
              [
                "Samoa",
                "Samoa"
              ],
              [
                "San Marino",
                "San Marino"
              ],
              [
                "Sao Tome and Principe",
                "Sao Tome and Principe"
              ],
              [
                "Saudi Arabia",
                "Saudi Arabia"
              ],
              [
                "Senegal",
                "Senegal"
              ],
              [
                "Serbia",
                "Serbia"
              ],
              [
                "Seychelles",
                "Seychelles"
              ],
              [
                "Sierra Leone",
                "Sierra Leone"
              ],
              [
                "Singapore",
                "Singapore"
              ],
              [
                "Slovakia",
                "Slovakia"
              ],
              [
                "Slovenia",
                "Slovenia"
              ],
              [
                "Solomon Islands",
                "Solomon Islands"
              ],
              [
                "Somalia",
                "Somalia"
              ],
              [
                "Somaliland",
                "Somaliland"
              ],
              [
                "South Africa",
                "South Africa"
              ],
              [
                "South Ossetia",
                "South Ossetia"
              ],
              [
                "South Sudan",
                "South Sudan"
              ],
              [
                "Spain",
                "Spain"
              ],
              [
                "Sri Lanka",
                "Sri Lanka"
              ],
              [
                "Sudan",
                "Sudan"
              ],
              [
                "Suriname",
                "Suriname"
              ],
              [
                "Svalbard",
                "Svalbard"
              ],
              [
                "eSwatini",
                "eSwatini"
              ],
              [
                "Sweden",
                "Sweden"
              ],
              [
                "Switzerland",
                "Switzerland"
              ],
              [
                "Syria",
                "Syria"
              ],
              [
                "Taiwan",
                "Taiwan"
              ],
              [
                "Tajikistan",
                "Tajikistan"
              ],
              [
                "Tanzania",
                "Tanzania"
              ],
              [
                "Thailand",
                "Thailand"
              ],
              [
                "Timor-Leste",
                "Timor-Leste"
              ],
              [
                "Togo",
                "Togo"
              ],
              [
                "Tokelau",
                "Tokelau"
              ],
              [
                "Tonga",
                "Tonga"
              ],
              [
                "Transnistria Pridnestrovie",
                "Transnistria Pridnestrovie"
              ],
              [
                "Trinidad and Tobago",
                "Trinidad and Tobago"
              ],
              [
                "Tristan da Cunha",
                "Tristan da Cunha"
              ],
              [
                "Tunisia",
                "Tunisia"
              ],
              [
                "Turkey",
                "Turkey"
              ],
              [
                "Turkmenistan",
                "Turkmenistan"
              ],
              [
                "Turks and Caicos Islands",
                "Turks and Caicos Islands"
              ],
              [
                "Tuvalu",
                "Tuvalu"
              ],
              [
                "Uganda",
                "Uganda"
              ],
              [
                "Ukraine",
                "Ukraine"
              ],
              [
                "United Arab Emirates",
                "United Arab Emirates"
              ],
              [
                "United Kingdom",
                "United Kingdom"
              ],
              [
                "Uruguay",
                "Uruguay"
              ],
              [
                "Uzbekistan",
                "Uzbekistan"
              ],
              [
                "Vanuatu",
                "Vanuatu"
              ],
              [
                "Vatican City",
                "Vatican City"
              ],
              [
                "Venezuela",
                "Venezuela"
              ],
              [
                "Vietnam",
                "Vietnam"
              ],
              [
                "British Virgin Islands",
                "British Virgin Islands"
              ],
              [
                "Isle of Man",
                "Isle of Man"
              ],
              [
                "US Virgin Islands",
                "US Virgin Islands"
              ],
              [
                "Wallis and Futuna",
                "Wallis and Futuna"
              ],
              [
                "Western Sahara",
                "Western Sahara"
              ],
              [
                "Yemen",
                "Yemen"
              ],
              [
                "Zambia",
                "Zambia"
              ],
              [
                "Zimbabwe",
                "Zimbabwe"
              ],
              [
                "other",
                "Other"
              ]
            ],
            "selected": [
              [
                "United States",
                "United States"
              ]
            ]
          },
          "validation": {}
        },
        {
          "selector": "#input_65_0",
          "id": "input_65_0",
          "name": "q65_canYou65",
          "type": "radio",
          "inputType": "radio",
          "labels": {
            "directLabel": "Yes",
            "groupLabel": "Are you authorized to work in the United States?",
            "placeholder": "",
            "contextText": "Yes"
          },
          "value": true,
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_65_1",
          "id": "input_65_1",
          "name": "q65_canYou65",
          "type": "radio",
          "inputType": "radio",
          "labels": {
            "directLabel": "No",
            "groupLabel": "Are you authorized to work in the United States?",
            "placeholder": "",
            "contextText": "No"
          },
          "value": false,
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_78_0",
          "id": "input_78_0",
          "name": "q78_selectThe[]",
          "type": "checkbox",
          "inputType": "checkbox",
          "labels": {
            "directLabel": "Monday",
            "groupLabel": "Select the day(s) you're available for work",
            "placeholder": "",
            "contextText": "Monday"
          },
          "value": true,
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_78_1",
          "id": "input_78_1",
          "name": "q78_selectThe[]",
          "type": "checkbox",
          "inputType": "checkbox",
          "labels": {
            "directLabel": "Tuesday",
            "groupLabel": "Select the day(s) you're available for work",
            "placeholder": "",
            "contextText": "Tuesday"
          },
          "value": true,
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_78_2",
          "id": "input_78_2",
          "name": "q78_selectThe[]",
          "type": "checkbox",
          "inputType": "checkbox",
          "labels": {
            "directLabel": "Wednesday",
            "groupLabel": "Select the day(s) you're available for work",
            "placeholder": "",
            "contextText": "Wednesday"
          },
          "value": true,
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_78_3",
          "id": "input_78_3",
          "name": "q78_selectThe[]",
          "type": "checkbox",
          "inputType": "checkbox",
          "labels": {
            "directLabel": "Thursday",
            "groupLabel": "Select the day(s) you're available for work",
            "placeholder": "",
            "contextText": "Thursday"
          },
          "value": true,
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_78_4",
          "id": "input_78_4",
          "name": "q78_selectThe[]",
          "type": "checkbox",
          "inputType": "checkbox",
          "labels": {
            "directLabel": "Friday",
            "groupLabel": "Select the day(s) you're available for work",
            "placeholder": "",
            "contextText": "Friday"
          },
          "value": true,
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_79",
          "id": "input_79",
          "name": "q79_listYour",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "List your available hours for work",
            "placeholder": ""
          },
          "value": "9 AM - 5 PM",
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        }
      ],
      "subsections": []
    },
    {
      "heading": "Education",
      "fields": [
        {
          "selector": "#input_64",
          "id": "input_64",
          "name": "q64_nameOf",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Name of College",
            "placeholder": " "
          },
          "value": "State University",
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_16_addr_line1",
          "id": "input_16_addr_line1",
          "name": "q16_schoolAddress[addr_line1]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "School Address",
            "placeholder": "",
            "contextText": "Street Address"
          },
          "value": "State University Campus, Building A, Room 101",
          "isEmpty": false,
          "isRequired": false,
          "validation": {
            "maxLength": 100
          }
        },
        {
          "selector": "#input_16_addr_line2",
          "id": "input_16_addr_line2",
          "name": "q16_schoolAddress[addr_line2]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Street Address Line 2",
            "placeholder": "",
            "contextText": "Street Address Line 2"
          },
          "value": "Room 101, Education City",
          "isEmpty": false,
          "isRequired": false,
          "validation": {
            "maxLength": 100
          }
        },
        {
          "selector": "#input_16_city",
          "id": "input_16_city",
          "name": "q16_schoolAddress[city]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "City",
            "placeholder": "",
            "contextText": "City"
          },
          "value": "Education City",
          "isEmpty": false,
          "isRequired": false,
          "validation": {
            "maxLength": 60
          }
        },
        {
          "selector": "#input_16_state",
          "id": "input_16_state",
          "name": "q16_schoolAddress[state]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "State / Province",
            "placeholder": "",
            "contextText": "State / Province"
          },
          "value": "California",
          "isEmpty": false,
          "isRequired": false,
          "validation": {
            "maxLength": 60
          }
        },
        {
          "selector": "#input_16_postal",
          "id": "input_16_postal",
          "name": "q16_schoolAddress[postal]",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Postal / Zip Code",
            "placeholder": "",
            "contextText": "Postal / Zip Code"
          },
          "value": "90210",
          "isEmpty": false,
          "isRequired": false,
          "validation": {
            "maxLength": 20
          }
        },
        {
          "selector": "#input_16_country",
          "id": "input_16_country",
          "name": "q16_schoolAddress[country]",
          "type": "select-one",
          "labels": {
            "directLabel": "Country",
            "contextText": "Country"
          },
          "value": "United States",
          "isEmpty": false,
          "isRequired": true,
          "options": {
            "unselected": [
              [
                "",
                "Please Select"
              ],
              [
                "Afghanistan",
                "Afghanistan"
              ],
              [
                "Albania",
                "Albania"
              ],
              [
                "Algeria",
                "Algeria"
              ],
              [
                "American Samoa",
                "American Samoa"
              ],
              [
                "Andorra",
                "Andorra"
              ],
              [
                "Angola",
                "Angola"
              ],
              [
                "Anguilla",
                "Anguilla"
              ],
              [
                "Antigua and Barbuda",
                "Antigua and Barbuda"
              ],
              [
                "Argentina",
                "Argentina"
              ],
              [
                "Armenia",
                "Armenia"
              ],
              [
                "Aruba",
                "Aruba"
              ],
              [
                "Australia",
                "Australia"
              ],
              [
                "Austria",
                "Austria"
              ],
              [
                "Azerbaijan",
                "Azerbaijan"
              ],
              [
                "The Bahamas",
                "The Bahamas"
              ],
              [
                "Bahrain",
                "Bahrain"
              ],
              [
                "Bangladesh",
                "Bangladesh"
              ],
              [
                "Barbados",
                "Barbados"
              ],
              [
                "Belarus",
                "Belarus"
              ],
              [
                "Belgium",
                "Belgium"
              ],
              [
                "Belize",
                "Belize"
              ],
              [
                "Benin",
                "Benin"
              ],
              [
                "Bermuda",
                "Bermuda"
              ],
              [
                "Bhutan",
                "Bhutan"
              ],
              [
                "Bolivia",
                "Bolivia"
              ],
              [
                "Bosnia and Herzegovina",
                "Bosnia and Herzegovina"
              ],
              [
                "Botswana",
                "Botswana"
              ],
              [
                "Brazil",
                "Brazil"
              ],
              [
                "Brunei",
                "Brunei"
              ],
              [
                "Bulgaria",
                "Bulgaria"
              ],
              [
                "Burkina Faso",
                "Burkina Faso"
              ],
              [
                "Burundi",
                "Burundi"
              ],
              [
                "Cambodia",
                "Cambodia"
              ],
              [
                "Cameroon",
                "Cameroon"
              ],
              [
                "Canada",
                "Canada"
              ],
              [
                "Cape Verde",
                "Cape Verde"
              ],
              [
                "Cayman Islands",
                "Cayman Islands"
              ],
              [
                "Central African Republic",
                "Central African Republic"
              ],
              [
                "Chad",
                "Chad"
              ],
              [
                "Chile",
                "Chile"
              ],
              [
                "China",
                "China"
              ],
              [
                "Christmas Island",
                "Christmas Island"
              ],
              [
                "Cocos (Keeling) Islands",
                "Cocos (Keeling) Islands"
              ],
              [
                "Colombia",
                "Colombia"
              ],
              [
                "Comoros",
                "Comoros"
              ],
              [
                "Congo",
                "Congo"
              ],
              [
                "Cook Islands",
                "Cook Islands"
              ],
              [
                "Costa Rica",
                "Costa Rica"
              ],
              [
                "Cote d'Ivoire",
                "Cote d'Ivoire"
              ],
              [
                "Croatia",
                "Croatia"
              ],
              [
                "Cuba",
                "Cuba"
              ],
              [
                "Curaçao",
                "Curaçao"
              ],
              [
                "Cyprus",
                "Cyprus"
              ],
              [
                "Czech Republic",
                "Czech Republic"
              ],
              [
                "Democratic Republic of the Congo",
                "Democratic Republic of the Congo"
              ],
              [
                "Denmark",
                "Denmark"
              ],
              [
                "Djibouti",
                "Djibouti"
              ],
              [
                "Dominica",
                "Dominica"
              ],
              [
                "Dominican Republic",
                "Dominican Republic"
              ],
              [
                "Ecuador",
                "Ecuador"
              ],
              [
                "Egypt",
                "Egypt"
              ],
              [
                "El Salvador",
                "El Salvador"
              ],
              [
                "Equatorial Guinea",
                "Equatorial Guinea"
              ],
              [
                "Eritrea",
                "Eritrea"
              ],
              [
                "Estonia",
                "Estonia"
              ],
              [
                "Ethiopia",
                "Ethiopia"
              ],
              [
                "Falkland Islands",
                "Falkland Islands"
              ],
              [
                "Faroe Islands",
                "Faroe Islands"
              ],
              [
                "Fiji",
                "Fiji"
              ],
              [
                "Finland",
                "Finland"
              ],
              [
                "France",
                "France"
              ],
              [
                "French Polynesia",
                "French Polynesia"
              ],
              [
                "Gabon",
                "Gabon"
              ],
              [
                "The Gambia",
                "The Gambia"
              ],
              [
                "Georgia",
                "Georgia"
              ],
              [
                "Germany",
                "Germany"
              ],
              [
                "Ghana",
                "Ghana"
              ],
              [
                "Gibraltar",
                "Gibraltar"
              ],
              [
                "Greece",
                "Greece"
              ],
              [
                "Greenland",
                "Greenland"
              ],
              [
                "Grenada",
                "Grenada"
              ],
              [
                "Guadeloupe",
                "Guadeloupe"
              ],
              [
                "Guam",
                "Guam"
              ],
              [
                "Guatemala",
                "Guatemala"
              ],
              [
                "Guernsey",
                "Guernsey"
              ],
              [
                "Guinea",
                "Guinea"
              ],
              [
                "Guinea-Bissau",
                "Guinea-Bissau"
              ],
              [
                "Guyana",
                "Guyana"
              ],
              [
                "Haiti",
                "Haiti"
              ],
              [
                "Honduras",
                "Honduras"
              ],
              [
                "Hong Kong",
                "Hong Kong"
              ],
              [
                "Hungary",
                "Hungary"
              ],
              [
                "Iceland",
                "Iceland"
              ],
              [
                "India",
                "India"
              ],
              [
                "Indonesia",
                "Indonesia"
              ],
              [
                "Iran",
                "Iran"
              ],
              [
                "Iraq",
                "Iraq"
              ],
              [
                "Ireland",
                "Ireland"
              ],
              [
                "Israel",
                "Israel"
              ],
              [
                "Italy",
                "Italy"
              ],
              [
                "Jamaica",
                "Jamaica"
              ],
              [
                "Japan",
                "Japan"
              ],
              [
                "Jersey",
                "Jersey"
              ],
              [
                "Jordan",
                "Jordan"
              ],
              [
                "Kazakhstan",
                "Kazakhstan"
              ],
              [
                "Kenya",
                "Kenya"
              ],
              [
                "Kiribati",
                "Kiribati"
              ],
              [
                "North Korea",
                "North Korea"
              ],
              [
                "South Korea",
                "South Korea"
              ],
              [
                "Kosovo",
                "Kosovo"
              ],
              [
                "Kuwait",
                "Kuwait"
              ],
              [
                "Kyrgyzstan",
                "Kyrgyzstan"
              ],
              [
                "Laos",
                "Laos"
              ],
              [
                "Latvia",
                "Latvia"
              ],
              [
                "Lebanon",
                "Lebanon"
              ],
              [
                "Lesotho",
                "Lesotho"
              ],
              [
                "Liberia",
                "Liberia"
              ],
              [
                "Libya",
                "Libya"
              ],
              [
                "Liechtenstein",
                "Liechtenstein"
              ],
              [
                "Lithuania",
                "Lithuania"
              ],
              [
                "Luxembourg",
                "Luxembourg"
              ],
              [
                "Macau",
                "Macau"
              ],
              [
                "Macedonia",
                "Macedonia"
              ],
              [
                "Madagascar",
                "Madagascar"
              ],
              [
                "Malawi",
                "Malawi"
              ],
              [
                "Malaysia",
                "Malaysia"
              ],
              [
                "Maldives",
                "Maldives"
              ],
              [
                "Mali",
                "Mali"
              ],
              [
                "Malta",
                "Malta"
              ],
              [
                "Marshall Islands",
                "Marshall Islands"
              ],
              [
                "Martinique",
                "Martinique"
              ],
              [
                "Mauritania",
                "Mauritania"
              ],
              [
                "Mauritius",
                "Mauritius"
              ],
              [
                "Mayotte",
                "Mayotte"
              ],
              [
                "Mexico",
                "Mexico"
              ],
              [
                "Micronesia",
                "Micronesia"
              ],
              [
                "Moldova",
                "Moldova"
              ],
              [
                "Monaco",
                "Monaco"
              ],
              [
                "Mongolia",
                "Mongolia"
              ],
              [
                "Montenegro",
                "Montenegro"
              ],
              [
                "Montserrat",
                "Montserrat"
              ],
              [
                "Morocco",
                "Morocco"
              ],
              [
                "Mozambique",
                "Mozambique"
              ],
              [
                "Myanmar",
                "Myanmar"
              ],
              [
                "Nagorno-Karabakh",
                "Nagorno-Karabakh"
              ],
              [
                "Namibia",
                "Namibia"
              ],
              [
                "Nauru",
                "Nauru"
              ],
              [
                "Nepal",
                "Nepal"
              ],
              [
                "Netherlands",
                "Netherlands"
              ],
              [
                "Netherlands Antilles",
                "Netherlands Antilles"
              ],
              [
                "New Caledonia",
                "New Caledonia"
              ],
              [
                "New Zealand",
                "New Zealand"
              ],
              [
                "Nicaragua",
                "Nicaragua"
              ],
              [
                "Niger",
                "Niger"
              ],
              [
                "Nigeria",
                "Nigeria"
              ],
              [
                "Niue",
                "Niue"
              ],
              [
                "Norfolk Island",
                "Norfolk Island"
              ],
              [
                "Turkish Republic of Northern Cyprus",
                "Turkish Republic of Northern Cyprus"
              ],
              [
                "Northern Mariana",
                "Northern Mariana"
              ],
              [
                "Norway",
                "Norway"
              ],
              [
                "Oman",
                "Oman"
              ],
              [
                "Pakistan",
                "Pakistan"
              ],
              [
                "Palau",
                "Palau"
              ],
              [
                "Palestine",
                "Palestine"
              ],
              [
                "Panama",
                "Panama"
              ],
              [
                "Papua New Guinea",
                "Papua New Guinea"
              ],
              [
                "Paraguay",
                "Paraguay"
              ],
              [
                "Peru",
                "Peru"
              ],
              [
                "Philippines",
                "Philippines"
              ],
              [
                "Pitcairn Islands",
                "Pitcairn Islands"
              ],
              [
                "Poland",
                "Poland"
              ],
              [
                "Portugal",
                "Portugal"
              ],
              [
                "Puerto Rico",
                "Puerto Rico"
              ],
              [
                "Qatar",
                "Qatar"
              ],
              [
                "Republic of the Congo",
                "Republic of the Congo"
              ],
              [
                "Romania",
                "Romania"
              ],
              [
                "Russia",
                "Russia"
              ],
              [
                "Rwanda",
                "Rwanda"
              ],
              [
                "Saint Barthelemy",
                "Saint Barthelemy"
              ],
              [
                "Saint Helena",
                "Saint Helena"
              ],
              [
                "Saint Kitts and Nevis",
                "Saint Kitts and Nevis"
              ],
              [
                "Saint Lucia",
                "Saint Lucia"
              ],
              [
                "Saint Martin",
                "Saint Martin"
              ],
              [
                "Saint Pierre and Miquelon",
                "Saint Pierre and Miquelon"
              ],
              [
                "Saint Vincent and the Grenadines",
                "Saint Vincent and the Grenadines"
              ],
              [
                "Samoa",
                "Samoa"
              ],
              [
                "San Marino",
                "San Marino"
              ],
              [
                "Sao Tome and Principe",
                "Sao Tome and Principe"
              ],
              [
                "Saudi Arabia",
                "Saudi Arabia"
              ],
              [
                "Senegal",
                "Senegal"
              ],
              [
                "Serbia",
                "Serbia"
              ],
              [
                "Seychelles",
                "Seychelles"
              ],
              [
                "Sierra Leone",
                "Sierra Leone"
              ],
              [
                "Singapore",
                "Singapore"
              ],
              [
                "Slovakia",
                "Slovakia"
              ],
              [
                "Slovenia",
                "Slovenia"
              ],
              [
                "Solomon Islands",
                "Solomon Islands"
              ],
              [
                "Somalia",
                "Somalia"
              ],
              [
                "Somaliland",
                "Somaliland"
              ],
              [
                "South Africa",
                "South Africa"
              ],
              [
                "South Ossetia",
                "South Ossetia"
              ],
              [
                "South Sudan",
                "South Sudan"
              ],
              [
                "Spain",
                "Spain"
              ],
              [
                "Sri Lanka",
                "Sri Lanka"
              ],
              [
                "Sudan",
                "Sudan"
              ],
              [
                "Suriname",
                "Suriname"
              ],
              [
                "Svalbard",
                "Svalbard"
              ],
              [
                "eSwatini",
                "eSwatini"
              ],
              [
                "Sweden",
                "Sweden"
              ],
              [
                "Switzerland",
                "Switzerland"
              ],
              [
                "Syria",
                "Syria"
              ],
              [
                "Taiwan",
                "Taiwan"
              ],
              [
                "Tajikistan",
                "Tajikistan"
              ],
              [
                "Tanzania",
                "Tanzania"
              ],
              [
                "Thailand",
                "Thailand"
              ],
              [
                "Timor-Leste",
                "Timor-Leste"
              ],
              [
                "Togo",
                "Togo"
              ],
              [
                "Tokelau",
                "Tokelau"
              ],
              [
                "Tonga",
                "Tonga"
              ],
              [
                "Transnistria Pridnestrovie",
                "Transnistria Pridnestrovie"
              ],
              [
                "Trinidad and Tobago",
                "Trinidad and Tobago"
              ],
              [
                "Tristan da Cunha",
                "Tristan da Cunha"
              ],
              [
                "Tunisia",
                "Tunisia"
              ],
              [
                "Turkey",
                "Turkey"
              ],
              [
                "Turkmenistan",
                "Turkmenistan"
              ],
              [
                "Turks and Caicos Islands",
                "Turks and Caicos Islands"
              ],
              [
                "Tuvalu",
                "Tuvalu"
              ],
              [
                "Uganda",
                "Uganda"
              ],
              [
                "Ukraine",
                "Ukraine"
              ],
              [
                "United Arab Emirates",
                "United Arab Emirates"
              ],
              [
                "United Kingdom",
                "United Kingdom"
              ],
              [
                "Uruguay",
                "Uruguay"
              ],
              [
                "Uzbekistan",
                "Uzbekistan"
              ],
              [
                "Vanuatu",
                "Vanuatu"
              ],
              [
                "Vatican City",
                "Vatican City"
              ],
              [
                "Venezuela",
                "Venezuela"
              ],
              [
                "Vietnam",
                "Vietnam"
              ],
              [
                "British Virgin Islands",
                "British Virgin Islands"
              ],
              [
                "Isle of Man",
                "Isle of Man"
              ],
              [
                "US Virgin Islands",
                "US Virgin Islands"
              ],
              [
                "Wallis and Futuna",
                "Wallis and Futuna"
              ],
              [
                "Western Sahara",
                "Western Sahara"
              ],
              [
                "Yemen",
                "Yemen"
              ],
              [
                "Zambia",
                "Zambia"
              ],
              [
                "Zimbabwe",
                "Zimbabwe"
              ],
              [
                "other",
                "Other"
              ]
            ],
            "selected": [
              [
                "United States",
                "United States"
              ]
            ]
          },
          "validation": {}
        },
        {
          "selector": "#input_56",
          "id": "input_56",
          "name": "q56_major",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Major",
            "placeholder": " "
          },
          "value": "Computer Science",
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_68",
          "id": "input_68",
          "name": "q68_year",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Year",
            "placeholder": ""
          },
          "value": "2026",
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_60",
          "id": "input_60",
          "name": "q60_otherColleges60",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Other College(s) info",
            "placeholder": " "
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_24",
          "id": "input_24",
          "name": "q24_highSchool",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "High School - City - Dates",
            "placeholder": " "
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        }
      ],
      "subsections": []
    },
    {
      "heading": "Employment",
      "fields": [
        {
          "selector": "#input_44",
          "id": "input_44",
          "name": "q44_place44",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Place",
            "placeholder": " "
          },
          "value": "Tech Corp",
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#month_69",
          "id": "month_69",
          "name": "q69_date[month]",
          "type": "tel",
          "inputType": "tel",
          "labels": {
            "directLabel": "Month",
            "placeholder": "",
            "contextText": "-Month"
          },
          "value": "11",
          "isEmpty": false,
          "isRequired": false,
          "validation": {
            "maxLength": 2
          }
        },
        {
          "selector": "#day_69",
          "id": "day_69",
          "name": "q69_date[day]",
          "type": "tel",
          "inputType": "tel",
          "labels": {
            "directLabel": "Day",
            "placeholder": "",
            "contextText": "-Day"
          },
          "value": "12",
          "isEmpty": false,
          "isRequired": false,
          "validation": {
            "maxLength": 2
          }
        },
        {
          "selector": "#year_69",
          "id": "year_69",
          "name": "q69_date[year]",
          "type": "tel",
          "inputType": "tel",
          "labels": {
            "directLabel": "Year",
            "placeholder": "",
            "contextText": "Year"
          },
          "value": "1942",
          "isEmpty": false,
          "isRequired": false,
          "validation": {
            "maxLength": 4
          }
        },
        {
          "selector": "#lite_mode_69",
          "id": "lite_mode_69",
          "name": "",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Date",
            "placeholder": "MM-DD-YYYY",
            "contextText": "Date"
          },
          "value": "11-12-1942",
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_70",
          "id": "input_70",
          "name": "q70_title70",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Title",
            "placeholder": ""
          },
          "value": "Senior Developer",
          "isEmpty": false,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_45",
          "id": "input_45",
          "name": "q45_place",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Place",
            "placeholder": " "
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#month_72",
          "id": "month_72",
          "name": "q72_date72[month]",
          "type": "tel",
          "inputType": "tel",
          "labels": {
            "directLabel": "Month",
            "placeholder": "",
            "contextText": "-Month"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {
            "maxLength": 2
          }
        },
        {
          "selector": "#day_72",
          "id": "day_72",
          "name": "q72_date72[day]",
          "type": "tel",
          "inputType": "tel",
          "labels": {
            "directLabel": "Day",
            "placeholder": "",
            "contextText": "-Day"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {
            "maxLength": 2
          }
        },
        {
          "selector": "#year_72",
          "id": "year_72",
          "name": "q72_date72[year]",
          "type": "tel",
          "inputType": "tel",
          "labels": {
            "directLabel": "Year",
            "placeholder": "",
            "contextText": "Year"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {
            "maxLength": 4
          }
        },
        {
          "selector": "#lite_mode_72",
          "id": "lite_mode_72",
          "name": "",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Date",
            "placeholder": "MM-DD-YYYY",
            "contextText": "Date"
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_73",
          "id": "input_73",
          "name": "q73_title",
          "type": "text",
          "inputType": "text",
          "labels": {
            "directLabel": "Title",
            "placeholder": ""
          },
          "value": "",
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_76",
          "id": "input_76",
          "name": "file",
          "type": "file",
          "inputType": "file",
          "labels": {
            "directLabel": "Attach Your CV",
            "placeholder": "",
            "precedingLabels": [
              "Drag and drop files here",
              "Choose a file",
              "Drag and drop files here",
              "Choose a file"
            ]
          },
          "value": null,
          "isEmpty": true,
          "isRequired": false,
          "validation": {}
        },
        {
          "selector": "#input_77",
          "id": "input_77",
          "name": "file",
          "type": "file",
          "inputType": "file",
          "labels": {
            "directLabel": "Attach Cover Letter",
            "placeholder": "",
            "precedingLabels": [
              "Drag and drop files here",
              "Choose a file",
              "Drag and drop files here",
              "Choose a file"
            ]
          },
          "value": null,
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
      "selector": "#first_12",
      "id": "first_12",
      "name": "q12_fullName12[first]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Full Name",
        "placeholder": "",
        "contextText": "First Name"
      },
      "value": "John",
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#last_12",
      "id": "last_12",
      "name": "q12_fullName12[last]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Last Name",
        "placeholder": "",
        "contextText": "Last Name"
      },
      "value": "Doe",
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_54",
      "id": "input_54",
      "name": "q54_preferredName54",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Preferred Name",
        "placeholder": " "
      },
      "value": "Johnny",
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_20_0",
      "id": "input_20_0",
      "name": "q20_gender20[]",
      "type": "checkbox",
      "inputType": "checkbox",
      "labels": {
        "directLabel": "Male",
        "groupLabel": "Gender",
        "placeholder": "",
        "contextText": "Male"
      },
      "value": true,
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_20_1",
      "id": "input_20_1",
      "name": "q20_gender20[]",
      "type": "checkbox",
      "inputType": "checkbox",
      "labels": {
        "directLabel": "Female",
        "groupLabel": "Gender",
        "placeholder": "",
        "contextText": "Female"
      },
      "value": false,
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_19_month",
      "id": "input_19_month",
      "name": "q19_birthDate19[month]",
      "type": "select-one",
      "labels": {
        "directLabel": "Month",
        "contextText": "Month"
      },
      "value": "11",
      "isEmpty": false,
      "isRequired": false,
      "options": {
        "unselected": [
          [
            "",
            "Please select a month"
          ],
          [
            "1",
            "January"
          ],
          [
            "2",
            "February"
          ],
          [
            "3",
            "March"
          ],
          [
            "4",
            "April"
          ],
          [
            "5",
            "May"
          ],
          [
            "6",
            "June"
          ],
          [
            "7",
            "July"
          ],
          [
            "8",
            "August"
          ],
          [
            "9",
            "September"
          ],
          [
            "10",
            "October"
          ],
          [
            "12",
            "December"
          ]
        ],
        "selected": [
          [
            "11",
            "November"
          ]
        ]
      },
      "validation": {}
    },
    {
      "selector": "#input_19_day",
      "id": "input_19_day",
      "name": "q19_birthDate19[day]",
      "type": "select-one",
      "labels": {
        "directLabel": "Day",
        "contextText": "Day"
      },
      "value": "18",
      "isEmpty": false,
      "isRequired": false,
      "options": {
        "unselected": [
          [
            "",
            "Please select a day"
          ],
          [
            "1",
            "1"
          ],
          [
            "2",
            "2"
          ],
          [
            "3",
            "3"
          ],
          [
            "4",
            "4"
          ],
          [
            "5",
            "5"
          ],
          [
            "6",
            "6"
          ],
          [
            "7",
            "7"
          ],
          [
            "8",
            "8"
          ],
          [
            "9",
            "9"
          ],
          [
            "10",
            "10"
          ],
          [
            "11",
            "11"
          ],
          [
            "12",
            "12"
          ],
          [
            "13",
            "13"
          ],
          [
            "14",
            "14"
          ],
          [
            "15",
            "15"
          ],
          [
            "16",
            "16"
          ],
          [
            "17",
            "17"
          ],
          [
            "19",
            "19"
          ],
          [
            "20",
            "20"
          ],
          [
            "21",
            "21"
          ],
          [
            "22",
            "22"
          ],
          [
            "23",
            "23"
          ],
          [
            "24",
            "24"
          ],
          [
            "25",
            "25"
          ],
          [
            "26",
            "26"
          ],
          [
            "27",
            "27"
          ],
          [
            "28",
            "28"
          ],
          [
            "29",
            "29"
          ],
          [
            "30",
            "30"
          ],
          [
            "31",
            "31"
          ]
        ],
        "selected": [
          [
            "18",
            "18"
          ]
        ]
      },
      "validation": {}
    },
    {
      "selector": "#input_19_year",
      "id": "input_19_year",
      "name": "q19_birthDate19[year]",
      "type": "select-one",
      "labels": {
        "directLabel": "Year",
        "contextText": "Year"
      },
      "value": "2004",
      "isEmpty": false,
      "isRequired": false,
      "options": {
        "unselected": [
          [
            "",
            "Please select a year"
          ],
          [
            "2025",
            "2025"
          ],
          [
            "2024",
            "2024"
          ],
          [
            "2023",
            "2023"
          ],
          [
            "2022",
            "2022"
          ],
          [
            "2021",
            "2021"
          ],
          [
            "2020",
            "2020"
          ],
          [
            "2019",
            "2019"
          ],
          [
            "2018",
            "2018"
          ],
          [
            "2017",
            "2017"
          ],
          [
            "2016",
            "2016"
          ],
          [
            "2015",
            "2015"
          ],
          [
            "2014",
            "2014"
          ],
          [
            "2013",
            "2013"
          ],
          [
            "2012",
            "2012"
          ],
          [
            "2011",
            "2011"
          ],
          [
            "2010",
            "2010"
          ],
          [
            "2009",
            "2009"
          ],
          [
            "2008",
            "2008"
          ],
          [
            "2007",
            "2007"
          ],
          [
            "2006",
            "2006"
          ],
          [
            "2005",
            "2005"
          ],
          [
            "2003",
            "2003"
          ],
          [
            "2002",
            "2002"
          ],
          [
            "2001",
            "2001"
          ],
          [
            "2000",
            "2000"
          ],
          [
            "1999",
            "1999"
          ],
          [
            "1998",
            "1998"
          ],
          [
            "1997",
            "1997"
          ],
          [
            "1996",
            "1996"
          ],
          [
            "1995",
            "1995"
          ],
          [
            "1994",
            "1994"
          ],
          [
            "1993",
            "1993"
          ],
          [
            "1992",
            "1992"
          ],
          [
            "1991",
            "1991"
          ],
          [
            "1990",
            "1990"
          ],
          [
            "1989",
            "1989"
          ],
          [
            "1988",
            "1988"
          ],
          [
            "1987",
            "1987"
          ],
          [
            "1986",
            "1986"
          ],
          [
            "1985",
            "1985"
          ],
          [
            "1984",
            "1984"
          ],
          [
            "1983",
            "1983"
          ],
          [
            "1982",
            "1982"
          ],
          [
            "1981",
            "1981"
          ],
          [
            "1980",
            "1980"
          ],
          [
            "1979",
            "1979"
          ],
          [
            "1978",
            "1978"
          ],
          [
            "1977",
            "1977"
          ],
          [
            "1976",
            "1976"
          ],
          [
            "1975",
            "1975"
          ],
          [
            "1974",
            "1974"
          ],
          [
            "1973",
            "1973"
          ],
          [
            "1972",
            "1972"
          ],
          [
            "1971",
            "1971"
          ],
          [
            "1970",
            "1970"
          ],
          [
            "1969",
            "1969"
          ],
          [
            "1968",
            "1968"
          ],
          [
            "1967",
            "1967"
          ],
          [
            "1966",
            "1966"
          ],
          [
            "1965",
            "1965"
          ],
          [
            "1964",
            "1964"
          ],
          [
            "1963",
            "1963"
          ],
          [
            "1962",
            "1962"
          ],
          [
            "1961",
            "1961"
          ],
          [
            "1960",
            "1960"
          ],
          [
            "1959",
            "1959"
          ],
          [
            "1958",
            "1958"
          ],
          [
            "1957",
            "1957"
          ],
          [
            "1956",
            "1956"
          ],
          [
            "1955",
            "1955"
          ],
          [
            "1954",
            "1954"
          ],
          [
            "1953",
            "1953"
          ],
          [
            "1952",
            "1952"
          ],
          [
            "1951",
            "1951"
          ],
          [
            "1950",
            "1950"
          ],
          [
            "1949",
            "1949"
          ],
          [
            "1948",
            "1948"
          ],
          [
            "1947",
            "1947"
          ],
          [
            "1946",
            "1946"
          ],
          [
            "1945",
            "1945"
          ],
          [
            "1944",
            "1944"
          ],
          [
            "1943",
            "1943"
          ],
          [
            "1942",
            "1942"
          ],
          [
            "1941",
            "1941"
          ],
          [
            "1940",
            "1940"
          ],
          [
            "1939",
            "1939"
          ],
          [
            "1938",
            "1938"
          ],
          [
            "1937",
            "1937"
          ],
          [
            "1936",
            "1936"
          ],
          [
            "1935",
            "1935"
          ],
          [
            "1934",
            "1934"
          ],
          [
            "1933",
            "1933"
          ],
          [
            "1932",
            "1932"
          ],
          [
            "1931",
            "1931"
          ],
          [
            "1930",
            "1930"
          ],
          [
            "1929",
            "1929"
          ],
          [
            "1928",
            "1928"
          ],
          [
            "1927",
            "1927"
          ],
          [
            "1926",
            "1926"
          ],
          [
            "1925",
            "1925"
          ],
          [
            "1924",
            "1924"
          ],
          [
            "1923",
            "1923"
          ],
          [
            "1922",
            "1922"
          ],
          [
            "1921",
            "1921"
          ],
          [
            "1920",
            "1920"
          ]
        ],
        "selected": [
          [
            "2004",
            "2004"
          ]
        ]
      },
      "validation": {}
    },
    {
      "selector": "#input_55",
      "id": "input_55",
      "name": "q55_age",
      "type": "number",
      "inputType": "number",
      "labels": {
        "directLabel": "Age",
        "placeholder": ""
      },
      "value": "20",
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_13",
      "id": "input_13",
      "name": "q13_email13",
      "type": "email",
      "inputType": "email",
      "labels": {
        "directLabel": "E-mail",
        "placeholder": "ex: myname@example.com",
        "contextText": "example@example.com"
      },
      "value": "john.doe@example.com",
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_17_full",
      "id": "input_17_full",
      "name": "q17_homeNumber[full]",
      "type": "tel",
      "inputType": "tel",
      "labels": {
        "directLabel": "Home Number",
        "placeholder": ""
      },
      "value": "(180) 052-3430",
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_18_full",
      "id": "input_18_full",
      "name": "q18_cellNumber[full]",
      "type": "tel",
      "inputType": "tel",
      "labels": {
        "directLabel": "Cell Number",
        "placeholder": ""
      },
      "value": "(933) 108-7882",
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_15_addr_line1",
      "id": "input_15_addr_line1",
      "name": "q15_permanentAddress[addr_line1]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Permanent Address",
        "placeholder": "",
        "contextText": "Street Address"
      },
      "value": "AXBX BUILDING, 3rd floor Room no 389",
      "isEmpty": false,
      "isRequired": false,
      "validation": {
        "maxLength": 100
      }
    },
    {
      "selector": "#input_15_addr_line2",
      "id": "input_15_addr_line2",
      "name": "q15_permanentAddress[addr_line2]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Street Address Line 2",
        "placeholder": "",
        "contextText": "Street Address Line 2"
      },
      "value": "PBS colony XYZ road, Nagole",
      "isEmpty": false,
      "isRequired": false,
      "validation": {
        "maxLength": 100
      }
    },
    {
      "selector": "#input_15_city",
      "id": "input_15_city",
      "name": "q15_permanentAddress[city]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "City",
        "placeholder": "",
        "contextText": "City"
      },
      "value": "Nagole",
      "isEmpty": false,
      "isRequired": false,
      "validation": {
        "maxLength": 60
      }
    },
    {
      "selector": "#input_15_state",
      "id": "input_15_state",
      "name": "q15_permanentAddress[state]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "State / Province",
        "placeholder": "",
        "contextText": "State / Province"
      },
      "value": "San Francisco",
      "isEmpty": false,
      "isRequired": false,
      "validation": {
        "maxLength": 60
      }
    },
    {
      "selector": "#input_15_postal",
      "id": "input_15_postal",
      "name": "q15_permanentAddress[postal]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Postal / Zip Code",
        "placeholder": "",
        "contextText": "Postal / Zip Code"
      },
      "value": "807101",
      "isEmpty": false,
      "isRequired": false,
      "validation": {
        "maxLength": 20
      }
    },
    {
      "selector": "#input_15_country",
      "id": "input_15_country",
      "name": "q15_permanentAddress[country]",
      "type": "select-one",
      "labels": {
        "directLabel": "Country",
        "contextText": "Country"
      },
      "value": "United States",
      "isEmpty": false,
      "isRequired": true,
      "options": {
        "unselected": [
          [
            "",
            "Please Select"
          ],
          [
            "Afghanistan",
            "Afghanistan"
          ],
          [
            "Albania",
            "Albania"
          ],
          [
            "Algeria",
            "Algeria"
          ],
          [
            "American Samoa",
            "American Samoa"
          ],
          [
            "Andorra",
            "Andorra"
          ],
          [
            "Angola",
            "Angola"
          ],
          [
            "Anguilla",
            "Anguilla"
          ],
          [
            "Antigua and Barbuda",
            "Antigua and Barbuda"
          ],
          [
            "Argentina",
            "Argentina"
          ],
          [
            "Armenia",
            "Armenia"
          ],
          [
            "Aruba",
            "Aruba"
          ],
          [
            "Australia",
            "Australia"
          ],
          [
            "Austria",
            "Austria"
          ],
          [
            "Azerbaijan",
            "Azerbaijan"
          ],
          [
            "The Bahamas",
            "The Bahamas"
          ],
          [
            "Bahrain",
            "Bahrain"
          ],
          [
            "Bangladesh",
            "Bangladesh"
          ],
          [
            "Barbados",
            "Barbados"
          ],
          [
            "Belarus",
            "Belarus"
          ],
          [
            "Belgium",
            "Belgium"
          ],
          [
            "Belize",
            "Belize"
          ],
          [
            "Benin",
            "Benin"
          ],
          [
            "Bermuda",
            "Bermuda"
          ],
          [
            "Bhutan",
            "Bhutan"
          ],
          [
            "Bolivia",
            "Bolivia"
          ],
          [
            "Bosnia and Herzegovina",
            "Bosnia and Herzegovina"
          ],
          [
            "Botswana",
            "Botswana"
          ],
          [
            "Brazil",
            "Brazil"
          ],
          [
            "Brunei",
            "Brunei"
          ],
          [
            "Bulgaria",
            "Bulgaria"
          ],
          [
            "Burkina Faso",
            "Burkina Faso"
          ],
          [
            "Burundi",
            "Burundi"
          ],
          [
            "Cambodia",
            "Cambodia"
          ],
          [
            "Cameroon",
            "Cameroon"
          ],
          [
            "Canada",
            "Canada"
          ],
          [
            "Cape Verde",
            "Cape Verde"
          ],
          [
            "Cayman Islands",
            "Cayman Islands"
          ],
          [
            "Central African Republic",
            "Central African Republic"
          ],
          [
            "Chad",
            "Chad"
          ],
          [
            "Chile",
            "Chile"
          ],
          [
            "China",
            "China"
          ],
          [
            "Christmas Island",
            "Christmas Island"
          ],
          [
            "Cocos (Keeling) Islands",
            "Cocos (Keeling) Islands"
          ],
          [
            "Colombia",
            "Colombia"
          ],
          [
            "Comoros",
            "Comoros"
          ],
          [
            "Congo",
            "Congo"
          ],
          [
            "Cook Islands",
            "Cook Islands"
          ],
          [
            "Costa Rica",
            "Costa Rica"
          ],
          [
            "Cote d'Ivoire",
            "Cote d'Ivoire"
          ],
          [
            "Croatia",
            "Croatia"
          ],
          [
            "Cuba",
            "Cuba"
          ],
          [
            "Curaçao",
            "Curaçao"
          ],
          [
            "Cyprus",
            "Cyprus"
          ],
          [
            "Czech Republic",
            "Czech Republic"
          ],
          [
            "Democratic Republic of the Congo",
            "Democratic Republic of the Congo"
          ],
          [
            "Denmark",
            "Denmark"
          ],
          [
            "Djibouti",
            "Djibouti"
          ],
          [
            "Dominica",
            "Dominica"
          ],
          [
            "Dominican Republic",
            "Dominican Republic"
          ],
          [
            "Ecuador",
            "Ecuador"
          ],
          [
            "Egypt",
            "Egypt"
          ],
          [
            "El Salvador",
            "El Salvador"
          ],
          [
            "Equatorial Guinea",
            "Equatorial Guinea"
          ],
          [
            "Eritrea",
            "Eritrea"
          ],
          [
            "Estonia",
            "Estonia"
          ],
          [
            "Ethiopia",
            "Ethiopia"
          ],
          [
            "Falkland Islands",
            "Falkland Islands"
          ],
          [
            "Faroe Islands",
            "Faroe Islands"
          ],
          [
            "Fiji",
            "Fiji"
          ],
          [
            "Finland",
            "Finland"
          ],
          [
            "France",
            "France"
          ],
          [
            "French Polynesia",
            "French Polynesia"
          ],
          [
            "Gabon",
            "Gabon"
          ],
          [
            "The Gambia",
            "The Gambia"
          ],
          [
            "Georgia",
            "Georgia"
          ],
          [
            "Germany",
            "Germany"
          ],
          [
            "Ghana",
            "Ghana"
          ],
          [
            "Gibraltar",
            "Gibraltar"
          ],
          [
            "Greece",
            "Greece"
          ],
          [
            "Greenland",
            "Greenland"
          ],
          [
            "Grenada",
            "Grenada"
          ],
          [
            "Guadeloupe",
            "Guadeloupe"
          ],
          [
            "Guam",
            "Guam"
          ],
          [
            "Guatemala",
            "Guatemala"
          ],
          [
            "Guernsey",
            "Guernsey"
          ],
          [
            "Guinea",
            "Guinea"
          ],
          [
            "Guinea-Bissau",
            "Guinea-Bissau"
          ],
          [
            "Guyana",
            "Guyana"
          ],
          [
            "Haiti",
            "Haiti"
          ],
          [
            "Honduras",
            "Honduras"
          ],
          [
            "Hong Kong",
            "Hong Kong"
          ],
          [
            "Hungary",
            "Hungary"
          ],
          [
            "Iceland",
            "Iceland"
          ],
          [
            "India",
            "India"
          ],
          [
            "Indonesia",
            "Indonesia"
          ],
          [
            "Iran",
            "Iran"
          ],
          [
            "Iraq",
            "Iraq"
          ],
          [
            "Ireland",
            "Ireland"
          ],
          [
            "Israel",
            "Israel"
          ],
          [
            "Italy",
            "Italy"
          ],
          [
            "Jamaica",
            "Jamaica"
          ],
          [
            "Japan",
            "Japan"
          ],
          [
            "Jersey",
            "Jersey"
          ],
          [
            "Jordan",
            "Jordan"
          ],
          [
            "Kazakhstan",
            "Kazakhstan"
          ],
          [
            "Kenya",
            "Kenya"
          ],
          [
            "Kiribati",
            "Kiribati"
          ],
          [
            "North Korea",
            "North Korea"
          ],
          [
            "South Korea",
            "South Korea"
          ],
          [
            "Kosovo",
            "Kosovo"
          ],
          [
            "Kuwait",
            "Kuwait"
          ],
          [
            "Kyrgyzstan",
            "Kyrgyzstan"
          ],
          [
            "Laos",
            "Laos"
          ],
          [
            "Latvia",
            "Latvia"
          ],
          [
            "Lebanon",
            "Lebanon"
          ],
          [
            "Lesotho",
            "Lesotho"
          ],
          [
            "Liberia",
            "Liberia"
          ],
          [
            "Libya",
            "Libya"
          ],
          [
            "Liechtenstein",
            "Liechtenstein"
          ],
          [
            "Lithuania",
            "Lithuania"
          ],
          [
            "Luxembourg",
            "Luxembourg"
          ],
          [
            "Macau",
            "Macau"
          ],
          [
            "Macedonia",
            "Macedonia"
          ],
          [
            "Madagascar",
            "Madagascar"
          ],
          [
            "Malawi",
            "Malawi"
          ],
          [
            "Malaysia",
            "Malaysia"
          ],
          [
            "Maldives",
            "Maldives"
          ],
          [
            "Mali",
            "Mali"
          ],
          [
            "Malta",
            "Malta"
          ],
          [
            "Marshall Islands",
            "Marshall Islands"
          ],
          [
            "Martinique",
            "Martinique"
          ],
          [
            "Mauritania",
            "Mauritania"
          ],
          [
            "Mauritius",
            "Mauritius"
          ],
          [
            "Mayotte",
            "Mayotte"
          ],
          [
            "Mexico",
            "Mexico"
          ],
          [
            "Micronesia",
            "Micronesia"
          ],
          [
            "Moldova",
            "Moldova"
          ],
          [
            "Monaco",
            "Monaco"
          ],
          [
            "Mongolia",
            "Mongolia"
          ],
          [
            "Montenegro",
            "Montenegro"
          ],
          [
            "Montserrat",
            "Montserrat"
          ],
          [
            "Morocco",
            "Morocco"
          ],
          [
            "Mozambique",
            "Mozambique"
          ],
          [
            "Myanmar",
            "Myanmar"
          ],
          [
            "Nagorno-Karabakh",
            "Nagorno-Karabakh"
          ],
          [
            "Namibia",
            "Namibia"
          ],
          [
            "Nauru",
            "Nauru"
          ],
          [
            "Nepal",
            "Nepal"
          ],
          [
            "Netherlands",
            "Netherlands"
          ],
          [
            "Netherlands Antilles",
            "Netherlands Antilles"
          ],
          [
            "New Caledonia",
            "New Caledonia"
          ],
          [
            "New Zealand",
            "New Zealand"
          ],
          [
            "Nicaragua",
            "Nicaragua"
          ],
          [
            "Niger",
            "Niger"
          ],
          [
            "Nigeria",
            "Nigeria"
          ],
          [
            "Niue",
            "Niue"
          ],
          [
            "Norfolk Island",
            "Norfolk Island"
          ],
          [
            "Turkish Republic of Northern Cyprus",
            "Turkish Republic of Northern Cyprus"
          ],
          [
            "Northern Mariana",
            "Northern Mariana"
          ],
          [
            "Norway",
            "Norway"
          ],
          [
            "Oman",
            "Oman"
          ],
          [
            "Pakistan",
            "Pakistan"
          ],
          [
            "Palau",
            "Palau"
          ],
          [
            "Palestine",
            "Palestine"
          ],
          [
            "Panama",
            "Panama"
          ],
          [
            "Papua New Guinea",
            "Papua New Guinea"
          ],
          [
            "Paraguay",
            "Paraguay"
          ],
          [
            "Peru",
            "Peru"
          ],
          [
            "Philippines",
            "Philippines"
          ],
          [
            "Pitcairn Islands",
            "Pitcairn Islands"
          ],
          [
            "Poland",
            "Poland"
          ],
          [
            "Portugal",
            "Portugal"
          ],
          [
            "Puerto Rico",
            "Puerto Rico"
          ],
          [
            "Qatar",
            "Qatar"
          ],
          [
            "Republic of the Congo",
            "Republic of the Congo"
          ],
          [
            "Romania",
            "Romania"
          ],
          [
            "Russia",
            "Russia"
          ],
          [
            "Rwanda",
            "Rwanda"
          ],
          [
            "Saint Barthelemy",
            "Saint Barthelemy"
          ],
          [
            "Saint Helena",
            "Saint Helena"
          ],
          [
            "Saint Kitts and Nevis",
            "Saint Kitts and Nevis"
          ],
          [
            "Saint Lucia",
            "Saint Lucia"
          ],
          [
            "Saint Martin",
            "Saint Martin"
          ],
          [
            "Saint Pierre and Miquelon",
            "Saint Pierre and Miquelon"
          ],
          [
            "Saint Vincent and the Grenadines",
            "Saint Vincent and the Grenadines"
          ],
          [
            "Samoa",
            "Samoa"
          ],
          [
            "San Marino",
            "San Marino"
          ],
          [
            "Sao Tome and Principe",
            "Sao Tome and Principe"
          ],
          [
            "Saudi Arabia",
            "Saudi Arabia"
          ],
          [
            "Senegal",
            "Senegal"
          ],
          [
            "Serbia",
            "Serbia"
          ],
          [
            "Seychelles",
            "Seychelles"
          ],
          [
            "Sierra Leone",
            "Sierra Leone"
          ],
          [
            "Singapore",
            "Singapore"
          ],
          [
            "Slovakia",
            "Slovakia"
          ],
          [
            "Slovenia",
            "Slovenia"
          ],
          [
            "Solomon Islands",
            "Solomon Islands"
          ],
          [
            "Somalia",
            "Somalia"
          ],
          [
            "Somaliland",
            "Somaliland"
          ],
          [
            "South Africa",
            "South Africa"
          ],
          [
            "South Ossetia",
            "South Ossetia"
          ],
          [
            "South Sudan",
            "South Sudan"
          ],
          [
            "Spain",
            "Spain"
          ],
          [
            "Sri Lanka",
            "Sri Lanka"
          ],
          [
            "Sudan",
            "Sudan"
          ],
          [
            "Suriname",
            "Suriname"
          ],
          [
            "Svalbard",
            "Svalbard"
          ],
          [
            "eSwatini",
            "eSwatini"
          ],
          [
            "Sweden",
            "Sweden"
          ],
          [
            "Switzerland",
            "Switzerland"
          ],
          [
            "Syria",
            "Syria"
          ],
          [
            "Taiwan",
            "Taiwan"
          ],
          [
            "Tajikistan",
            "Tajikistan"
          ],
          [
            "Tanzania",
            "Tanzania"
          ],
          [
            "Thailand",
            "Thailand"
          ],
          [
            "Timor-Leste",
            "Timor-Leste"
          ],
          [
            "Togo",
            "Togo"
          ],
          [
            "Tokelau",
            "Tokelau"
          ],
          [
            "Tonga",
            "Tonga"
          ],
          [
            "Transnistria Pridnestrovie",
            "Transnistria Pridnestrovie"
          ],
          [
            "Trinidad and Tobago",
            "Trinidad and Tobago"
          ],
          [
            "Tristan da Cunha",
            "Tristan da Cunha"
          ],
          [
            "Tunisia",
            "Tunisia"
          ],
          [
            "Turkey",
            "Turkey"
          ],
          [
            "Turkmenistan",
            "Turkmenistan"
          ],
          [
            "Turks and Caicos Islands",
            "Turks and Caicos Islands"
          ],
          [
            "Tuvalu",
            "Tuvalu"
          ],
          [
            "Uganda",
            "Uganda"
          ],
          [
            "Ukraine",
            "Ukraine"
          ],
          [
            "United Arab Emirates",
            "United Arab Emirates"
          ],
          [
            "United Kingdom",
            "United Kingdom"
          ],
          [
            "Uruguay",
            "Uruguay"
          ],
          [
            "Uzbekistan",
            "Uzbekistan"
          ],
          [
            "Vanuatu",
            "Vanuatu"
          ],
          [
            "Vatican City",
            "Vatican City"
          ],
          [
            "Venezuela",
            "Venezuela"
          ],
          [
            "Vietnam",
            "Vietnam"
          ],
          [
            "British Virgin Islands",
            "British Virgin Islands"
          ],
          [
            "Isle of Man",
            "Isle of Man"
          ],
          [
            "US Virgin Islands",
            "US Virgin Islands"
          ],
          [
            "Wallis and Futuna",
            "Wallis and Futuna"
          ],
          [
            "Western Sahara",
            "Western Sahara"
          ],
          [
            "Yemen",
            "Yemen"
          ],
          [
            "Zambia",
            "Zambia"
          ],
          [
            "Zimbabwe",
            "Zimbabwe"
          ],
          [
            "other",
            "Other"
          ]
        ],
        "selected": [
          [
            "United States",
            "United States"
          ]
        ]
      },
      "validation": {}
    },
    {
      "selector": "#input_65_0",
      "id": "input_65_0",
      "name": "q65_canYou65",
      "type": "radio",
      "inputType": "radio",
      "labels": {
        "directLabel": "Yes",
        "groupLabel": "Are you authorized to work in the United States?",
        "placeholder": "",
        "contextText": "Yes"
      },
      "value": true,
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_65_1",
      "id": "input_65_1",
      "name": "q65_canYou65",
      "type": "radio",
      "inputType": "radio",
      "labels": {
        "directLabel": "No",
        "groupLabel": "Are you authorized to work in the United States?",
        "placeholder": "",
        "contextText": "No"
      },
      "value": false,
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_78_0",
      "id": "input_78_0",
      "name": "q78_selectThe[]",
      "type": "checkbox",
      "inputType": "checkbox",
      "labels": {
        "directLabel": "Monday",
        "groupLabel": "Select the day(s) you're available for work",
        "placeholder": "",
        "contextText": "Monday"
      },
      "value": true,
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_78_1",
      "id": "input_78_1",
      "name": "q78_selectThe[]",
      "type": "checkbox",
      "inputType": "checkbox",
      "labels": {
        "directLabel": "Tuesday",
        "groupLabel": "Select the day(s) you're available for work",
        "placeholder": "",
        "contextText": "Tuesday"
      },
      "value": true,
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_78_2",
      "id": "input_78_2",
      "name": "q78_selectThe[]",
      "type": "checkbox",
      "inputType": "checkbox",
      "labels": {
        "directLabel": "Wednesday",
        "groupLabel": "Select the day(s) you're available for work",
        "placeholder": "",
        "contextText": "Wednesday"
      },
      "value": true,
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_78_3",
      "id": "input_78_3",
      "name": "q78_selectThe[]",
      "type": "checkbox",
      "inputType": "checkbox",
      "labels": {
        "directLabel": "Thursday",
        "groupLabel": "Select the day(s) you're available for work",
        "placeholder": "",
        "contextText": "Thursday"
      },
      "value": true,
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_78_4",
      "id": "input_78_4",
      "name": "q78_selectThe[]",
      "type": "checkbox",
      "inputType": "checkbox",
      "labels": {
        "directLabel": "Friday",
        "groupLabel": "Select the day(s) you're available for work",
        "placeholder": "",
        "contextText": "Friday"
      },
      "value": true,
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_79",
      "id": "input_79",
      "name": "q79_listYour",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "List your available hours for work",
        "placeholder": ""
      },
      "value": "9 AM - 5 PM",
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_64",
      "id": "input_64",
      "name": "q64_nameOf",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Name of College",
        "placeholder": " "
      },
      "value": "State University",
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_16_addr_line1",
      "id": "input_16_addr_line1",
      "name": "q16_schoolAddress[addr_line1]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "School Address",
        "placeholder": "",
        "contextText": "Street Address"
      },
      "value": "State University Campus, Building A, Room 101",
      "isEmpty": false,
      "isRequired": false,
      "validation": {
        "maxLength": 100
      }
    },
    {
      "selector": "#input_16_addr_line2",
      "id": "input_16_addr_line2",
      "name": "q16_schoolAddress[addr_line2]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Street Address Line 2",
        "placeholder": "",
        "contextText": "Street Address Line 2"
      },
      "value": "Room 101, Education City",
      "isEmpty": false,
      "isRequired": false,
      "validation": {
        "maxLength": 100
      }
    },
    {
      "selector": "#input_16_city",
      "id": "input_16_city",
      "name": "q16_schoolAddress[city]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "City",
        "placeholder": "",
        "contextText": "City"
      },
      "value": "Education City",
      "isEmpty": false,
      "isRequired": false,
      "validation": {
        "maxLength": 60
      }
    },
    {
      "selector": "#input_16_state",
      "id": "input_16_state",
      "name": "q16_schoolAddress[state]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "State / Province",
        "placeholder": "",
        "contextText": "State / Province"
      },
      "value": "California",
      "isEmpty": false,
      "isRequired": false,
      "validation": {
        "maxLength": 60
      }
    },
    {
      "selector": "#input_16_postal",
      "id": "input_16_postal",
      "name": "q16_schoolAddress[postal]",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Postal / Zip Code",
        "placeholder": "",
        "contextText": "Postal / Zip Code"
      },
      "value": "90210",
      "isEmpty": false,
      "isRequired": false,
      "validation": {
        "maxLength": 20
      }
    },
    {
      "selector": "#input_16_country",
      "id": "input_16_country",
      "name": "q16_schoolAddress[country]",
      "type": "select-one",
      "labels": {
        "directLabel": "Country",
        "contextText": "Country"
      },
      "value": "United States",
      "isEmpty": false,
      "isRequired": true,
      "options": {
        "unselected": [
          [
            "",
            "Please Select"
          ],
          [
            "Afghanistan",
            "Afghanistan"
          ],
          [
            "Albania",
            "Albania"
          ],
          [
            "Algeria",
            "Algeria"
          ],
          [
            "American Samoa",
            "American Samoa"
          ],
          [
            "Andorra",
            "Andorra"
          ],
          [
            "Angola",
            "Angola"
          ],
          [
            "Anguilla",
            "Anguilla"
          ],
          [
            "Antigua and Barbuda",
            "Antigua and Barbuda"
          ],
          [
            "Argentina",
            "Argentina"
          ],
          [
            "Armenia",
            "Armenia"
          ],
          [
            "Aruba",
            "Aruba"
          ],
          [
            "Australia",
            "Australia"
          ],
          [
            "Austria",
            "Austria"
          ],
          [
            "Azerbaijan",
            "Azerbaijan"
          ],
          [
            "The Bahamas",
            "The Bahamas"
          ],
          [
            "Bahrain",
            "Bahrain"
          ],
          [
            "Bangladesh",
            "Bangladesh"
          ],
          [
            "Barbados",
            "Barbados"
          ],
          [
            "Belarus",
            "Belarus"
          ],
          [
            "Belgium",
            "Belgium"
          ],
          [
            "Belize",
            "Belize"
          ],
          [
            "Benin",
            "Benin"
          ],
          [
            "Bermuda",
            "Bermuda"
          ],
          [
            "Bhutan",
            "Bhutan"
          ],
          [
            "Bolivia",
            "Bolivia"
          ],
          [
            "Bosnia and Herzegovina",
            "Bosnia and Herzegovina"
          ],
          [
            "Botswana",
            "Botswana"
          ],
          [
            "Brazil",
            "Brazil"
          ],
          [
            "Brunei",
            "Brunei"
          ],
          [
            "Bulgaria",
            "Bulgaria"
          ],
          [
            "Burkina Faso",
            "Burkina Faso"
          ],
          [
            "Burundi",
            "Burundi"
          ],
          [
            "Cambodia",
            "Cambodia"
          ],
          [
            "Cameroon",
            "Cameroon"
          ],
          [
            "Canada",
            "Canada"
          ],
          [
            "Cape Verde",
            "Cape Verde"
          ],
          [
            "Cayman Islands",
            "Cayman Islands"
          ],
          [
            "Central African Republic",
            "Central African Republic"
          ],
          [
            "Chad",
            "Chad"
          ],
          [
            "Chile",
            "Chile"
          ],
          [
            "China",
            "China"
          ],
          [
            "Christmas Island",
            "Christmas Island"
          ],
          [
            "Cocos (Keeling) Islands",
            "Cocos (Keeling) Islands"
          ],
          [
            "Colombia",
            "Colombia"
          ],
          [
            "Comoros",
            "Comoros"
          ],
          [
            "Congo",
            "Congo"
          ],
          [
            "Cook Islands",
            "Cook Islands"
          ],
          [
            "Costa Rica",
            "Costa Rica"
          ],
          [
            "Cote d'Ivoire",
            "Cote d'Ivoire"
          ],
          [
            "Croatia",
            "Croatia"
          ],
          [
            "Cuba",
            "Cuba"
          ],
          [
            "Curaçao",
            "Curaçao"
          ],
          [
            "Cyprus",
            "Cyprus"
          ],
          [
            "Czech Republic",
            "Czech Republic"
          ],
          [
            "Democratic Republic of the Congo",
            "Democratic Republic of the Congo"
          ],
          [
            "Denmark",
            "Denmark"
          ],
          [
            "Djibouti",
            "Djibouti"
          ],
          [
            "Dominica",
            "Dominica"
          ],
          [
            "Dominican Republic",
            "Dominican Republic"
          ],
          [
            "Ecuador",
            "Ecuador"
          ],
          [
            "Egypt",
            "Egypt"
          ],
          [
            "El Salvador",
            "El Salvador"
          ],
          [
            "Equatorial Guinea",
            "Equatorial Guinea"
          ],
          [
            "Eritrea",
            "Eritrea"
          ],
          [
            "Estonia",
            "Estonia"
          ],
          [
            "Ethiopia",
            "Ethiopia"
          ],
          [
            "Falkland Islands",
            "Falkland Islands"
          ],
          [
            "Faroe Islands",
            "Faroe Islands"
          ],
          [
            "Fiji",
            "Fiji"
          ],
          [
            "Finland",
            "Finland"
          ],
          [
            "France",
            "France"
          ],
          [
            "French Polynesia",
            "French Polynesia"
          ],
          [
            "Gabon",
            "Gabon"
          ],
          [
            "The Gambia",
            "The Gambia"
          ],
          [
            "Georgia",
            "Georgia"
          ],
          [
            "Germany",
            "Germany"
          ],
          [
            "Ghana",
            "Ghana"
          ],
          [
            "Gibraltar",
            "Gibraltar"
          ],
          [
            "Greece",
            "Greece"
          ],
          [
            "Greenland",
            "Greenland"
          ],
          [
            "Grenada",
            "Grenada"
          ],
          [
            "Guadeloupe",
            "Guadeloupe"
          ],
          [
            "Guam",
            "Guam"
          ],
          [
            "Guatemala",
            "Guatemala"
          ],
          [
            "Guernsey",
            "Guernsey"
          ],
          [
            "Guinea",
            "Guinea"
          ],
          [
            "Guinea-Bissau",
            "Guinea-Bissau"
          ],
          [
            "Guyana",
            "Guyana"
          ],
          [
            "Haiti",
            "Haiti"
          ],
          [
            "Honduras",
            "Honduras"
          ],
          [
            "Hong Kong",
            "Hong Kong"
          ],
          [
            "Hungary",
            "Hungary"
          ],
          [
            "Iceland",
            "Iceland"
          ],
          [
            "India",
            "India"
          ],
          [
            "Indonesia",
            "Indonesia"
          ],
          [
            "Iran",
            "Iran"
          ],
          [
            "Iraq",
            "Iraq"
          ],
          [
            "Ireland",
            "Ireland"
          ],
          [
            "Israel",
            "Israel"
          ],
          [
            "Italy",
            "Italy"
          ],
          [
            "Jamaica",
            "Jamaica"
          ],
          [
            "Japan",
            "Japan"
          ],
          [
            "Jersey",
            "Jersey"
          ],
          [
            "Jordan",
            "Jordan"
          ],
          [
            "Kazakhstan",
            "Kazakhstan"
          ],
          [
            "Kenya",
            "Kenya"
          ],
          [
            "Kiribati",
            "Kiribati"
          ],
          [
            "North Korea",
            "North Korea"
          ],
          [
            "South Korea",
            "South Korea"
          ],
          [
            "Kosovo",
            "Kosovo"
          ],
          [
            "Kuwait",
            "Kuwait"
          ],
          [
            "Kyrgyzstan",
            "Kyrgyzstan"
          ],
          [
            "Laos",
            "Laos"
          ],
          [
            "Latvia",
            "Latvia"
          ],
          [
            "Lebanon",
            "Lebanon"
          ],
          [
            "Lesotho",
            "Lesotho"
          ],
          [
            "Liberia",
            "Liberia"
          ],
          [
            "Libya",
            "Libya"
          ],
          [
            "Liechtenstein",
            "Liechtenstein"
          ],
          [
            "Lithuania",
            "Lithuania"
          ],
          [
            "Luxembourg",
            "Luxembourg"
          ],
          [
            "Macau",
            "Macau"
          ],
          [
            "Macedonia",
            "Macedonia"
          ],
          [
            "Madagascar",
            "Madagascar"
          ],
          [
            "Malawi",
            "Malawi"
          ],
          [
            "Malaysia",
            "Malaysia"
          ],
          [
            "Maldives",
            "Maldives"
          ],
          [
            "Mali",
            "Mali"
          ],
          [
            "Malta",
            "Malta"
          ],
          [
            "Marshall Islands",
            "Marshall Islands"
          ],
          [
            "Martinique",
            "Martinique"
          ],
          [
            "Mauritania",
            "Mauritania"
          ],
          [
            "Mauritius",
            "Mauritius"
          ],
          [
            "Mayotte",
            "Mayotte"
          ],
          [
            "Mexico",
            "Mexico"
          ],
          [
            "Micronesia",
            "Micronesia"
          ],
          [
            "Moldova",
            "Moldova"
          ],
          [
            "Monaco",
            "Monaco"
          ],
          [
            "Mongolia",
            "Mongolia"
          ],
          [
            "Montenegro",
            "Montenegro"
          ],
          [
            "Montserrat",
            "Montserrat"
          ],
          [
            "Morocco",
            "Morocco"
          ],
          [
            "Mozambique",
            "Mozambique"
          ],
          [
            "Myanmar",
            "Myanmar"
          ],
          [
            "Nagorno-Karabakh",
            "Nagorno-Karabakh"
          ],
          [
            "Namibia",
            "Namibia"
          ],
          [
            "Nauru",
            "Nauru"
          ],
          [
            "Nepal",
            "Nepal"
          ],
          [
            "Netherlands",
            "Netherlands"
          ],
          [
            "Netherlands Antilles",
            "Netherlands Antilles"
          ],
          [
            "New Caledonia",
            "New Caledonia"
          ],
          [
            "New Zealand",
            "New Zealand"
          ],
          [
            "Nicaragua",
            "Nicaragua"
          ],
          [
            "Niger",
            "Niger"
          ],
          [
            "Nigeria",
            "Nigeria"
          ],
          [
            "Niue",
            "Niue"
          ],
          [
            "Norfolk Island",
            "Norfolk Island"
          ],
          [
            "Turkish Republic of Northern Cyprus",
            "Turkish Republic of Northern Cyprus"
          ],
          [
            "Northern Mariana",
            "Northern Mariana"
          ],
          [
            "Norway",
            "Norway"
          ],
          [
            "Oman",
            "Oman"
          ],
          [
            "Pakistan",
            "Pakistan"
          ],
          [
            "Palau",
            "Palau"
          ],
          [
            "Palestine",
            "Palestine"
          ],
          [
            "Panama",
            "Panama"
          ],
          [
            "Papua New Guinea",
            "Papua New Guinea"
          ],
          [
            "Paraguay",
            "Paraguay"
          ],
          [
            "Peru",
            "Peru"
          ],
          [
            "Philippines",
            "Philippines"
          ],
          [
            "Pitcairn Islands",
            "Pitcairn Islands"
          ],
          [
            "Poland",
            "Poland"
          ],
          [
            "Portugal",
            "Portugal"
          ],
          [
            "Puerto Rico",
            "Puerto Rico"
          ],
          [
            "Qatar",
            "Qatar"
          ],
          [
            "Republic of the Congo",
            "Republic of the Congo"
          ],
          [
            "Romania",
            "Romania"
          ],
          [
            "Russia",
            "Russia"
          ],
          [
            "Rwanda",
            "Rwanda"
          ],
          [
            "Saint Barthelemy",
            "Saint Barthelemy"
          ],
          [
            "Saint Helena",
            "Saint Helena"
          ],
          [
            "Saint Kitts and Nevis",
            "Saint Kitts and Nevis"
          ],
          [
            "Saint Lucia",
            "Saint Lucia"
          ],
          [
            "Saint Martin",
            "Saint Martin"
          ],
          [
            "Saint Pierre and Miquelon",
            "Saint Pierre and Miquelon"
          ],
          [
            "Saint Vincent and the Grenadines",
            "Saint Vincent and the Grenadines"
          ],
          [
            "Samoa",
            "Samoa"
          ],
          [
            "San Marino",
            "San Marino"
          ],
          [
            "Sao Tome and Principe",
            "Sao Tome and Principe"
          ],
          [
            "Saudi Arabia",
            "Saudi Arabia"
          ],
          [
            "Senegal",
            "Senegal"
          ],
          [
            "Serbia",
            "Serbia"
          ],
          [
            "Seychelles",
            "Seychelles"
          ],
          [
            "Sierra Leone",
            "Sierra Leone"
          ],
          [
            "Singapore",
            "Singapore"
          ],
          [
            "Slovakia",
            "Slovakia"
          ],
          [
            "Slovenia",
            "Slovenia"
          ],
          [
            "Solomon Islands",
            "Solomon Islands"
          ],
          [
            "Somalia",
            "Somalia"
          ],
          [
            "Somaliland",
            "Somaliland"
          ],
          [
            "South Africa",
            "South Africa"
          ],
          [
            "South Ossetia",
            "South Ossetia"
          ],
          [
            "South Sudan",
            "South Sudan"
          ],
          [
            "Spain",
            "Spain"
          ],
          [
            "Sri Lanka",
            "Sri Lanka"
          ],
          [
            "Sudan",
            "Sudan"
          ],
          [
            "Suriname",
            "Suriname"
          ],
          [
            "Svalbard",
            "Svalbard"
          ],
          [
            "eSwatini",
            "eSwatini"
          ],
          [
            "Sweden",
            "Sweden"
          ],
          [
            "Switzerland",
            "Switzerland"
          ],
          [
            "Syria",
            "Syria"
          ],
          [
            "Taiwan",
            "Taiwan"
          ],
          [
            "Tajikistan",
            "Tajikistan"
          ],
          [
            "Tanzania",
            "Tanzania"
          ],
          [
            "Thailand",
            "Thailand"
          ],
          [
            "Timor-Leste",
            "Timor-Leste"
          ],
          [
            "Togo",
            "Togo"
          ],
          [
            "Tokelau",
            "Tokelau"
          ],
          [
            "Tonga",
            "Tonga"
          ],
          [
            "Transnistria Pridnestrovie",
            "Transnistria Pridnestrovie"
          ],
          [
            "Trinidad and Tobago",
            "Trinidad and Tobago"
          ],
          [
            "Tristan da Cunha",
            "Tristan da Cunha"
          ],
          [
            "Tunisia",
            "Tunisia"
          ],
          [
            "Turkey",
            "Turkey"
          ],
          [
            "Turkmenistan",
            "Turkmenistan"
          ],
          [
            "Turks and Caicos Islands",
            "Turks and Caicos Islands"
          ],
          [
            "Tuvalu",
            "Tuvalu"
          ],
          [
            "Uganda",
            "Uganda"
          ],
          [
            "Ukraine",
            "Ukraine"
          ],
          [
            "United Arab Emirates",
            "United Arab Emirates"
          ],
          [
            "United Kingdom",
            "United Kingdom"
          ],
          [
            "Uruguay",
            "Uruguay"
          ],
          [
            "Uzbekistan",
            "Uzbekistan"
          ],
          [
            "Vanuatu",
            "Vanuatu"
          ],
          [
            "Vatican City",
            "Vatican City"
          ],
          [
            "Venezuela",
            "Venezuela"
          ],
          [
            "Vietnam",
            "Vietnam"
          ],
          [
            "British Virgin Islands",
            "British Virgin Islands"
          ],
          [
            "Isle of Man",
            "Isle of Man"
          ],
          [
            "US Virgin Islands",
            "US Virgin Islands"
          ],
          [
            "Wallis and Futuna",
            "Wallis and Futuna"
          ],
          [
            "Western Sahara",
            "Western Sahara"
          ],
          [
            "Yemen",
            "Yemen"
          ],
          [
            "Zambia",
            "Zambia"
          ],
          [
            "Zimbabwe",
            "Zimbabwe"
          ],
          [
            "other",
            "Other"
          ]
        ],
        "selected": [
          [
            "United States",
            "United States"
          ]
        ]
      },
      "validation": {}
    },
    {
      "selector": "#input_56",
      "id": "input_56",
      "name": "q56_major",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Major",
        "placeholder": " "
      },
      "value": "Computer Science",
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_68",
      "id": "input_68",
      "name": "q68_year",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Year",
        "placeholder": ""
      },
      "value": "2026",
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_60",
      "id": "input_60",
      "name": "q60_otherColleges60",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Other College(s) info",
        "placeholder": " "
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_24",
      "id": "input_24",
      "name": "q24_highSchool",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "High School - City - Dates",
        "placeholder": " "
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_44",
      "id": "input_44",
      "name": "q44_place44",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Place",
        "placeholder": " "
      },
      "value": "Tech Corp",
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#month_69",
      "id": "month_69",
      "name": "q69_date[month]",
      "type": "tel",
      "inputType": "tel",
      "labels": {
        "directLabel": "Month",
        "placeholder": "",
        "contextText": "-Month"
      },
      "value": "11",
      "isEmpty": false,
      "isRequired": false,
      "validation": {
        "maxLength": 2
      }
    },
    {
      "selector": "#day_69",
      "id": "day_69",
      "name": "q69_date[day]",
      "type": "tel",
      "inputType": "tel",
      "labels": {
        "directLabel": "Day",
        "placeholder": "",
        "contextText": "-Day"
      },
      "value": "12",
      "isEmpty": false,
      "isRequired": false,
      "validation": {
        "maxLength": 2
      }
    },
    {
      "selector": "#year_69",
      "id": "year_69",
      "name": "q69_date[year]",
      "type": "tel",
      "inputType": "tel",
      "labels": {
        "directLabel": "Year",
        "placeholder": "",
        "contextText": "Year"
      },
      "value": "1942",
      "isEmpty": false,
      "isRequired": false,
      "validation": {
        "maxLength": 4
      }
    },
    {
      "selector": "#lite_mode_69",
      "id": "lite_mode_69",
      "name": "",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Date",
        "placeholder": "MM-DD-YYYY",
        "contextText": "Date"
      },
      "value": "11-12-1942",
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_70",
      "id": "input_70",
      "name": "q70_title70",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Title",
        "placeholder": ""
      },
      "value": "Senior Developer",
      "isEmpty": false,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_45",
      "id": "input_45",
      "name": "q45_place",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Place",
        "placeholder": " "
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#month_72",
      "id": "month_72",
      "name": "q72_date72[month]",
      "type": "tel",
      "inputType": "tel",
      "labels": {
        "directLabel": "Month",
        "placeholder": "",
        "contextText": "-Month"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {
        "maxLength": 2
      }
    },
    {
      "selector": "#day_72",
      "id": "day_72",
      "name": "q72_date72[day]",
      "type": "tel",
      "inputType": "tel",
      "labels": {
        "directLabel": "Day",
        "placeholder": "",
        "contextText": "-Day"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {
        "maxLength": 2
      }
    },
    {
      "selector": "#year_72",
      "id": "year_72",
      "name": "q72_date72[year]",
      "type": "tel",
      "inputType": "tel",
      "labels": {
        "directLabel": "Year",
        "placeholder": "",
        "contextText": "Year"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {
        "maxLength": 4
      }
    },
    {
      "selector": "#lite_mode_72",
      "id": "lite_mode_72",
      "name": "",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Date",
        "placeholder": "MM-DD-YYYY",
        "contextText": "Date"
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_73",
      "id": "input_73",
      "name": "q73_title",
      "type": "text",
      "inputType": "text",
      "labels": {
        "directLabel": "Title",
        "placeholder": ""
      },
      "value": "",
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_76",
      "id": "input_76",
      "name": "file",
      "type": "file",
      "inputType": "file",
      "labels": {
        "directLabel": "Attach Your CV",
        "placeholder": "",
        "precedingLabels": [
          "Drag and drop files here",
          "Choose a file",
          "Drag and drop files here",
          "Choose a file"
        ]
      },
      "value": null,
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    },
    {
      "selector": "#input_77",
      "id": "input_77",
      "name": "file",
      "type": "file",
      "inputType": "file",
      "labels": {
        "directLabel": "Attach Cover Letter",
        "placeholder": "",
        "precedingLabels": [
          "Drag and drop files here",
          "Choose a file",
          "Drag and drop files here",
          "Choose a file"
        ]
      },
      "value": null,
      "isEmpty": true,
      "isRequired": false,
      "validation": {}
    }
  ],
  "metadata": {
    "totalFields": 51,
    "requiredFields": 2,
    "emptyFields": 12,
    "formAction": "https://submit.jotform.com/submit/252470591559465",
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
        output_file = './test_response_main/output_actions.json'
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


