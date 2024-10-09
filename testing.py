import streamlit as st
import requests
import json
import os

# Function to handle file saving with auto-incrementing filenames
def save_response_to_file(response_content):
    # Get the current files in the directory
    files = [f for f in os.listdir() if f.startswith('response_gpt') and f.endswith('.json')]
    
    # Find the highest file number
    if files:
        file_numbers = [int(f.split('_')[1].split('.')[0]) for f in files]
        new_file_number = max(file_numbers) + 1
    else:
        new_file_number = 1

    # Generate the new filename
    new_filename = f'response_{new_file_number}.json'
    
    # Save the response to the new file
    with open(new_filename, 'w') as file:
        json.dump(response_content, file, indent=4)
    
    return new_filename

# Streamlit app
st.title("Content Submission to API")

# Input text area
content = st.text_area("Enter your content here", height=300)

# API Key input
api_key = st.text_input("Enter your API key", type="password")

# Endpoint input
endpoint = st.text_input("Enter the API endpoint", "https://ao-openai-uat.openai.azure.com/openai/deployments/AOpfxm3/chat/completions?api-version=2024-02-15-preview")

# Submit button
if st.button("Submit"):
    if content and api_key:
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }

        # Define payload structure
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an AI assistant designed to generate comprehensive educational content based on specific topics. For each topic provided, you will generate various types of content as specified below..."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": content
                        }
                    ]
                }
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 4096
        }

        try:
            # Send POST request
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            
            # Save response to file
            filename = save_response_to_file(response.json())
            st.success(f"Response saved to {filename}")

        except requests.RequestException as e:
            st.error(f"Failed to make the request: {e}")
    else:
        st.error("Please enter both content and API key.")