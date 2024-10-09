import streamlit as st
import requests
import json
import os
from tkinter import Tk
from tkinter.filedialog import asksaveasfilename

# Function to handle file saving with user-defined directory
def save_response_to_file(response_content, model_name):
    # Initialize Tkinter root and hide the main window
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    # Ask user where to save the file
    filetypes = [('JSON files', '*.json')]
    initial_filename = f"response_{model_name}.json"
    save_path = asksaveasfilename(defaultextension=".json", filetypes=filetypes, initialfile=initial_filename)

    # Proceed only if a save path was specified
    if save_path:
        with open(save_path, 'w') as file:
            json.dump(response_content, file, indent=4)
        return save_path
    else:
        return None

# Streamlit app
st.title("Content Submission to API")

content = st.text_area("Enter your content here", height=300)
system_instruction = st.text_area("Enter the system instruction", "You are an AI assistant...", height=100)
api_key = st.text_input("Enter your API key", type="password")
model_selection = st.selectbox("Select the model", ["OpenAI GPT-4", "Gemini 1.5 Pro"])

if st.button("Submit"):
    if content and api_key:
        try:
            if model_selection == "OpenAI GPT-4":
                endpoint = "https://ao-openai-uat.openai.azure.com/openai/deployments/AOpfxm3/chat/completions?api-version=2024-02-15-preview"
                headers = {
                    "Content-Type": "application/json",
                    "api-key": api_key,
                }

                payload = {
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": content},
                    ],
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 4096,
                }

                response = requests.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                filename = save_response_to_file(response.json(), "OpenAI-GPT4")
                if filename:
                    st.success(f"Response saved to {filename}")
                else:
                    st.warning("File save was cancelled.")

            elif model_selection == "Gemini 1.5 Pro":
                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
                headers = {
                    "Content-Type": "application/json",
                }

                payload = {
                    "system_instruction": system_instruction,
                    "content": content,
                }

                response = requests.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                filename = save_response_to_file(response.json(), "Gemini-1.5-Pro")
                if filename:
                    st.success(f"Response saved to {filename}")
                else:
                    st.warning("File save was cancelled.")

        except requests.RequestException as e:
            st.error(f"Failed to make the request: {e}")
    else:
        st.error("Please enter content, API key.")