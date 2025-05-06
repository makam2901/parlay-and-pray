# src/backend/llm.py
import google.generativeai as genai
import os
from typing import List, Dict, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define keywords that trigger the redirection message
PREDICTION_KEYWORDS = ["predict", "generate", "suggest", "create", "build", "fantasy team", "dream11 team", "pick team"]

# The redirection message
REDIRECTION_MESSAGE = "To predict a fantasy team, please use the options on the left sidebar to set up your match details and click '🔮 Predict Team'."

def configure_gemini(api_key: str):
    """Configures the Generative AI client."""
    try:
        genai.configure(api_key=api_key)
        logging.info("Gemini AI configured successfully.")
    except Exception as e:
        logging.error(f"Error configuring Gemini AI: {e}", exc_info=True)
        raise

def format_history_for_gemini(history: List[Dict[str, str]]) -> List[Dict[str, Union[str, List[str]]]]:
    """
    Formats the chat history from {'role': 'user'/'model', 'content': 'message'}
    to Gemini's expected format {'role': 'user'/'model', 'parts': ['message']}.
    """
    gemini_history = []
    for message in history:
        role = message.get('role')
        content = message.get('content')
        if role and content:
            # Map 'model' role to 'model', keep 'user' as 'user'
            gemini_role = 'model' if role == 'assistant' else role # Adjust if your frontend uses 'assistant'
            gemini_history.append({"role": gemini_role, "parts": [content]})
    return gemini_history

def get_chat_response(api_key: str, user_message: str, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    """
    Gets a chat response from Gemini Pro, handles history, and redirects for predictions.

    Args:
        api_key: The Gemini API key.
        user_message: The user's latest message.
        history: The conversation history in {'role': 'user'/'model', 'content': 'message'} format.

    Returns:
        A tuple containing:
            - The response message (string).
            - The updated conversation history (list of dicts).
    """
    try:
        # 1. Check if the user is asking for a prediction
        lower_user_message = user_message.lower()
        if any(keyword in lower_user_message for keyword in PREDICTION_KEYWORDS):
            logging.info("Prediction request detected. Responding with redirection message.")
            response_message = REDIRECTION_MESSAGE
        else:
            # 2. Configure Gemini (might be redundant if called elsewhere, but safe)
            configure_gemini(api_key)

            # 3. Format history for Gemini
            gemini_history = format_history_for_gemini(history)
            logging.debug(f"Formatted Gemini History: {gemini_history}")

            # 4. Initialize the model and start chat
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            chat = model.start_chat(history=gemini_history)

            # 5. Send message and get response
            logging.info(f"Sending message to Gemini: {user_message}")
            response = chat.send_message(user_message)
            response_message = response.text
            logging.info(f"Received response from Gemini: {response_message[:100]}...") # Log truncated response

        # 6. Update history (local format)
        updated_history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response_message} # Use 'assistant' or 'model' consistently
        ]

        return response_message, updated_history

    except Exception as e:
        logging.error(f"Error getting chat response from Gemini: {e}", exc_info=True)
        # Provide a user-friendly error message
        error_message = "Sorry, I encountered an error trying to respond. Please try again later."
        # Return the error message but keep the history up to the point of failure
        updated_history = history + [
             {"role": "user", "content": user_message},
             {"role": "assistant", "content": error_message}
        ]
        return error_message, updated_history