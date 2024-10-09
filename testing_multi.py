import streamlit as st
import requests
import json
import os

# Function to handle file saving with auto-incrementing filenames
def save_response_to_file(response_content, file_name, model_name):
    # Sanitize the file name input
    base_file_name = f"{file_name}_{model_name}"

    # Get the current files in the directory
    files = [f for f in os.listdir() if f.startswith(base_file_name) and f.endswith('.json')]
    
    # Find the highest file number
    if files:
        file_numbers = [int(f.split('_')[-1].split('.')[0]) for f in files if f.split('_')[-1].split('.')[0].isdigit()]
        new_file_number = max(file_numbers) + 1 if file_numbers else 1
    else:
        new_file_number = 1

    # Generate the new filename
    new_filename = f'{base_file_name}_{new_file_number}.json'
    
    # Save the response to the new file
    with open(new_filename, 'w') as file:
        json.dump(response_content, file, indent=4)
    
    return new_filename

# Streamlit app
st.title("Content Submission to API")

# Input for content and system instructions
content = st.text_area("Enter your content here", height=300)
system_instruction = st.text_area("Enter the system instruction", "You are an AI assistant...", height=100)

# API Key input
api_key = st.text_input("Enter your API key", type="password")

# Model selection
model_selection = st.selectbox("Select the model", ["OpenAI GPT-4", "Gemini 1.5 Pro"])

# File name input
file_name = st.text_input("Enter the file name (without extension)", "response")

# Submit button
if st.button("Submit"):
    if content and api_key and file_name:
        if model_selection == "OpenAI GPT-4":
            # OpenAI GPT-4 Endpoint and payload
            endpoint = "https://ao-openai-uat.openai.azure.com/openai/deployments/AOpfxm3/chat/completions?api-version=2024-02-15-preview"
            headers = {
                "Content-Type": "application/json",
                "api-key": api_key,
            }

            payload = {
                "response_format":{ "type": "json_object" },
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system_instruction
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
                # Send request to OpenAI GPT-4
                response = requests.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                filename = save_response_to_file(response.json(), file_name, "OpenAI-GPT4")
                st.success(f"Response saved to {filename}")
            except requests.RequestException as e:
                st.error(f"Failed to make the request: {e}")

        elif model_selection == "Gemini 1.5 Pro":
            # Gemini API endpoint and payload
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
            headers = {
                "Content-Type": "application/json",
            }

            payload = {
                "system_instruction": {
                    "parts": {
                        "text": system_instruction
                    }
                },
                "contents": {
                    "parts": {
                        "text": content
                    }
                }
            }

            try:
                # Send request to Gemini API
                response = requests.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                filename = save_response_to_file(response.json(), file_name, "Gemini-1.5-Pro")
                st.success(f"Response saved to {filename}")
            except requests.RequestException as e:
                st.error(f"Failed to make the request: {e}")
    else:
        st.error("Please enter content, API key, and file name.")