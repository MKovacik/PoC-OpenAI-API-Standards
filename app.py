import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS package
from openai import AzureOpenAI
#import openai
from flasgger import swag_from, Swagger
import logging
import time
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()
CORS(app)  # Enable CORS for your Flask app and specify allowed origins
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/swagger/",
    "securityDefinitions": {
        "Bearer": {
            "type": "apiKey",
            "name": "Authorization",
            "in": "header",
            "description": "Enter your bearer token in the format **Bearer <token>**"
        }
    }
}
Swagger(app, config=swagger_config)  # This line will enable swagger with custom security definition

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

security_token = os.getenv("token")
# Example function to validate the API KEY
def is_valid_api_key(api_key):
    # Here, you would check the API key against your stored value(s)
    # For demonstration, let's assume a simple check against an environment variable
    expected_api_key = os.getenv("token", "").strip()  # Ensure default to empty string and strip whitespace
    return api_key.strip() == expected_api_key

client = AzureOpenAI(
  azure_endpoint = os.getenv("api_url"), 
  api_key= os.getenv("api_key"),  
  api_version="2024-02-15-preview"
)

# Define token limits for each model
model_token_limits = {
    'gpt-3.5-turbo': 4096,
    'gpt-4': 8192,  
    # Add more models and their token limits here
}

#Using ticktoken for tokenization will be better for the future
def calculate_token_usage(conversation):
    total_tokens = 0
    for exchange in conversation:
        # Counting tokens as the number of space-separated words
        prompt_length = len(exchange.get('prompt', ''))
        response_length = len(exchange.get('response', ''))
        total_tokens += prompt_length + response_length
    return total_tokens

def trim_conversation(conversation, model_name):
    token_limit = model_token_limits.get(model_name)
    if not token_limit:
        raise ValueError(f"Token limit not defined for model {model_name}")
    while calculate_token_usage(conversation) > token_limit:
        # Remove the oldest exchanges first to fit within the token limit
        conversation.pop(0)
    return conversation

conversations= {}

@app.route('/conversations/<session_id>', methods=['GET'])
@swag_from({
    'tags': ['Conversations'],
    'description': 'Get a conversation based on the session ID',
    'parameters': [
        {
            'name': 'session_id',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'The session ID to retrieve the conversation'
        }
    ],
    'responses': {
        200: {
            'description': 'Successful operation',
            'schema': {
                'type': 'object',
                'properties': {
                    'session_id': {'type': 'string', 'description': 'The session ID'},
                    'conversation': {'type': 'array', 'items': {'type': 'object'}, 'description': 'The conversation'}
                }
            }
        },
        404: {'description': 'Conversation not found for the given session ID'}
    }
})
def get_conversation(session_id):
    if session_id in conversations:
        conversation = conversations[session_id]
        return jsonify({"session_id": session_id, "conversation": conversation})
    else:
        return jsonify({"error": "Conversation not found for the given session ID"}), 404

@app.route('/chat/completions', methods=['POST'])
@swag_from({
    'tags': ['Chat Completions'],
    'description': 'Generate text completions based on a prompt',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'model': {
                        'type': 'string',
                        'description': 'Model to use for the completion. Defaults to "default-model"',
                        'default': 'default-model'
                    },
                    'messages': {
                        'type': 'array',
                        'description': 'Array of message objects',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'role': {
                                    'type': 'string',
                                    'description': 'Role of the message sender (e.g., "user", "system")'
                                },
                                'content': {
                                    'type': 'string',
                                    'description': 'Content of the message'
                                }
                            },
                            'required': ['role', 'content']
                        }
                    }
                },
                'required': ['model', 'messages']
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Successful operation',
            'schema': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'string', 'description': 'Unique ID for the response'},
                    'object': {'type': 'string', 'description': 'Object type, e.g., text_completion'},
                    'created': {'type': 'integer', 'description': 'Timestamp of creation'},
                    'model': {'type': 'string', 'description': 'Model used for the completion'},
                    'choices': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'text': {'type': 'string', 'description': 'Generated text'},
                                'index': {'type': 'integer', 'description': 'Index of the choice'},
                                'logprobs': {'type': 'null', 'description': 'Log probabilities (optional)'},
                                'finish_reason': {'type': 'string', 'description': 'Reason why the generation was finished'}
                            }
                        },
                        'description': 'Array of generated choices'
                    }
                }
            }
        },
        400: {'description': 'Invalid input'}
    },
    'security': [
        {"Bearer": []}  # References the security definition "Bearer"
    ]
}, validation=True)

def chat_completions():
    # Extract the API KEY from the Authorization header
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        api_key = auth_header[7:]  # Correctly strip 'Bearer ' prefix to get the actual API key
    else:
        return jsonify({"error": "Authorization header is missing or invalid"}), 401
    
    # Validate the API KEY
    if not is_valid_api_key(api_key):
        return jsonify({"error": "Invalid API key (header)" + api_key + " and is not matching to (token from env. variable) " + security_token}), 401
    
    request_data = request.json
    model = request_data.get('model', 'default-model')  # Default model if not specified
    
    # Adjusted to support 'messages' array in the request payload
    messages = request_data.get('messages')
    if not messages:
        return jsonify({"error": "Messages are required"}), 400
    
    # Extract 'prompt' and 'session_id' from the messages array if necessary
    prompt = next((msg['content'] for msg in messages if msg['role'] == 'user'), '')
    session_id = request_data.get('session_id', 'default-session')  # Example, adjust as needed

    # Check if the model is supported
    if model not in model_token_limits:
        return jsonify({"error": f"Model {model} is not supported"}), 400
    
    promtdefintion = "You are a helpful assistant."
    # Prepare the messages for the OpenAI API
    if session_id not in conversations:
        conversations[session_id] = [
            {"role": "system", "content": promtdefintion}
        ]
    messages = conversations[session_id]
    messages.append({"role": "user", "content": prompt})

    try:
        # Call the OpenAI API with the required arguments
        client = AzureOpenAI(
            azure_endpoint=os.getenv("api_url"), # Use the API URL obtained from the environment variable - Azure config
            api_key=os.getenv("api_key"),  # Use the API key obtained from the environment variable - Azure config
            api_version="2024-02-15-preview"
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
         # Process the response and format it accordingly
        generated_text = response.choices[0].message.content
        # Add the assistant's response to the conversation
        messages.append({"role": "assistant", "content": generated_text})
        messages = trim_conversation(messages, model)
        # Prepare the response data
        response_data = {
            "id": session_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "text": generated_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length"
                }
            ]
        }

        return jsonify(response_data)

    except Exception as e:
        error_message = str(e)
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
