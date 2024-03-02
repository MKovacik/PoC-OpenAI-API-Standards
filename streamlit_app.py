import os
from dotenv import load_dotenv
import streamlit as st
import requests
import json

# Load environment variables
load_dotenv()
# Streamlit interface
st.title('Chat with AI')

# Initialize or update session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

session_id = "1234"
token = os.getenv("token")
model = os.getenv("model")

# URLs or paths to avatar images
user_avatar = "icons/user_icon.png"
ai_avatar = "icons/ai_icon.png"

base_api_url = os.getenv("api_url") 
endpoint_path = "v1/chat/completions"
api_url = f"{base_api_url}/{endpoint_path}"  # Use f-string for clean concatenation


# Using a form for input and submit button
with st.form(key='message_form'):
    prompt = st.text_input('Say something to the AI:', '')
    submit_button = st.form_submit_button('Send')

if submit_button and prompt:
    response = requests.post(
        api_url,
        json={
            'model': model,
            'prompt': prompt,
            'session_id': session_id,
            'token': token
        })
    if response.status_code == 200:
        data = response.json()
        response_text = data['choices'][0]['text']
        st.session_state.history.append({"prompt": prompt, "response": response_text})
    else:
        st.error("Failed to get a response from the API")

# Conversation history
for exchange in st.session_state.history:
    col1, col2 = st.columns([0.1, 1], gap="small")
    with col1:
        st.image(user_avatar, width=30, output_format="PNG")
    with col2:
        st.write(f"{exchange['prompt']}")

    col1, col2 = st.columns([1, 0.1], gap="small")
    with col1:
        st.write(f"{exchange['response']}")
    with col2:
        st.image(ai_avatar, width=30, output_format="PNG")