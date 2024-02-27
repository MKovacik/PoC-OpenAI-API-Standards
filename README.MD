# AI Chat Application

This project is a Flask-based web application designed to facilitate conversations with an AI model. It allows users to send messages to the AI and receive responses, maintaining a conversation history. The application also integrates with an external AI service via API calls.

## Features

- Flask web server handling API requests and responses.
- Conversation history tracking.
- Integration with external AI services using API tokens for authentication.
- Swagger documentation for API endpoints.
- CORS enabled for cross-origin requests.
- Implementin Open AI API standars with Python openai library 1.0 and higher
- Streamlit interface for interactive user engagement (Example of consuming Open AI API standars)
- 
## Installation

To set up and run this project locally, follow these steps:

### Prerequisites

- Python 3.6 or later
- pip

### Steps

1. **Clone the Repository**

```
sh
git clone https://github.com/MKovacik/PoC---OpenAI-API-Standards.git
cd repo
```

2. **Install Dependencies**

   Install the required Python packages using `pip`:

```
sh
pip install -r requirements.txt
```

   This will install Flask, requests, python-dotenv, and other necessary libraries as specified in your `requirements.txt` file.

3. **Set Up Environment Variables**

   Create a `.env` file in the root directory of the project. Add your API token and other configuration variables like so:

```
api_key='your Open AI API key'
api_url='your Open AI API URL'
token='you can specify your token'
model='your model name'
```

   Make sure to replace 'your Open AI API key', 'your Open AI API URL' and 'your model name' with your actual information Key,URl,Model. 'you can specify your token' can be anything. 

4. **Run the Application**

   Start the Flask application by running:

```
sh
flask run
```

   or if you're using the application structure as in `app.py`:

```
sh
python app.py
```

   This will start the server on `localhost` with the default port `5000` (or another port if specified).

5. **Run the Streamlit Interface**

   To interact with the application through the Streamlit interface, run:

```
sh
streamlit run streamlit_app.py
```

## Usage

Once the application is running, you can access the Swagger UI to test the API endpoints by navigating to `http://localhost:5000/apidocs` in your web browser.

To interact with the AI, use the `/v1/chat/completions` endpoint to send messages and receive responses. For retrieving conversation history, use the `/v1/conversations/<session_id>` endpoint.

To interact with the AI through the Streamlit interface, ensure you've started the Streamlit server as described above. The interface allows for sending messages to the AI and viewing responses, along with conversation history. Implementing Streamlit interface is just PoC, that by implementation of Open AI API standars you can do integration with any other AI API and open-source projects or COTS products.
