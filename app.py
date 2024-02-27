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
CORS(app)  # Enable CORS for your Flask app and specify allowed origins
Swagger(app)  # This line will enable swagger
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

security_token = os.getenv("token")
client = AzureOpenAI(
  azure_endpoint = os.getenv("api_url"), 
  api_key= os.getenv("api_key"),  
  api_version="2024-02-15-preview"
)
clientdalle = AzureOpenAI(
    api_version="2023-12-01-preview",  
    azure_endpoint = os.getenv("api_url"), 
    api_key= os.getenv("api_key"), 
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
        prompt_tokens = len(exchange['prompt'].split())
        response_tokens = len(exchange['response'].split())
        total_tokens += prompt_tokens + response_tokens
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

@app.route('/v1/conversations/<session_id>', methods=['GET'])
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

@app.route('/v1/chat/completions', methods=['POST'])
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
                        'description': 'Model to use for the completion.',
                        'default': 'default-model'
                    },
                    'prompt': {
                        'type': 'string',
                        'description': 'Prompt to generate text from',
                        'required': True
                    },
                     'token': {
                        'type': 'string',
                        'description': 'Security token',
                    },
                     'session_id': {
                        'type': 'string',
                        'description': 'Unique session ID for the conversation',
                        'required': True
                    }
                }
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
    }
}, validation=True)
def chat_completions():
    request_data = request.json
    model = request_data.get('model', 'default-model')  # Default model if not specified
    prompt = request_data.get('prompt', '')
    session_id = request_data.get('session_id')
    token = request_data.get('token')

    if token[:8] != security_token or not prompt or not session_id:
        return jsonify({"error": "Invalid token"}), 401
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
        conversation = trim_conversation(conversation, model)
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

@app.route('/api/dalle', methods=['POST'])
@swag_from({
    'tags': ['DALL-E'],
    'description': '!! Currently not working due the API standarts missmatch. A DALL-E endpoint to generate images from text descriptions',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'id': 'TextDescription',
                'required': ['prompt', 'n', 'resolution', 'token'],
                'properties': {
                    'prompt': {
                        'type': 'string',
                        'description': 'The text description to generate an image from'
                    },
                    'n': {
                        'type': 'integer',
                        'description': 'The number of images to generate'
                    },
                    'resolution': {
                        'type': 'string',
                        'description': 'The resolution of the generated images'
                    },
                    'token': {
                        'type': 'string',
                        'description': 'Token used for authentication'
                    }
                }
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Successful operation'
        },
        400: {
            'description': 'Invalid input'
        }
    }
})
def dalle():
    user_input = request.json.get('prompt')
    image_n = request.json.get('n', 1)  # Default to 1 if not specified
    resolution = request.json.get('resolution')
    token = request.json.get('token')

    if token[:8] != security_token:
        return jsonify({"error": "Invalid token"}), 401

    try:
        response = clientdalle.images.generate(
          prompt=user_input,
          size=resolution,
          n=image_n
        )

        json_response = json.loads(response.model_dump_json())
        image_url = json_response["data"][0]["url"]

        if not image_url:
            raise ValueError("No image URL in response")

        return jsonify({"image_url": image_url})
        #image_url = response["data"][0]["url"]

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5001, debug=True)
