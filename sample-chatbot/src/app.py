# src/app.py
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_community.chat_models.ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# --- Configuration ---
# Use environment variables or a config file for better practice
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") # Default Ollama URL
SCORING_API_URL = os.getenv("SCORING_API_URL", "http://scoring:8001/score") # URL for the *new* scoring API

# --- Initialize LLM and Memory ---
# We initialize memory here, but it needs to be managed per-session/user in a real app
# For simplicity here, we'll re-initialize per request path, which isn't ideal for true memory.
# A better approach involves session management (e.g., using user IDs and storing memory objects).
llm = ChatOllama(model="mistral", base_url=OLLAMA_BASE_URL)

# --- Pydantic Models for Request/Response ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    chat_history: List[Dict[str, str]] # Return the updated history

# --- FastAPI App ---
app = FastAPI()

# In-memory store for conversation state (replace with DB/Redis for production)
conversation_state = {}

# --- Helper Function for Prediction Logic ---
def handle_prediction_flow(user_id: str, latest_user_msg: str):
    """
    Checks for prediction intent or if user provided data.
    Returns:
        - Dict with prompt message if intent detected.
        - Dict with prediction message if data is valid and scoring succeeds.
        - None otherwise.
    """
    prediction_keywords = ["predict", "check", "disease", "heart", "diagnose", "symptoms"]
    state = conversation_state.get(user_id, {"prediction_mode": False})

    # 1. Check if user wants to predict
    if any(keyword in latest_user_msg.lower() for keyword in prediction_keywords) and not state["prediction_mode"]:
        state["prediction_mode"] = True
        conversation_state[user_id] = state
        return {
            "type": "prompt",
            "message": (
                "Okay, I can help with a preliminary check based on common factors. "
                "Please provide the following 13 values as a single comma-separated list:\n"
                "1. Age (years)\n2. Sex (1=male, 0=female)\n3. Chest pain type (0-3)\n4. Resting blood pressure (mm Hg)\n"
                "5. Cholesterol level (mg/dl)\n6. Fasting blood sugar > 120 mg/dl (1=true, 0=false)\n7. Resting ECG results (0-2)\n8. Max heart rate achieved (thalach)\n"
                "9. Exercise-induced angina (exang: 1=yes, 0=no)\n10. ST depression induced by exercise relative to rest (oldpeak)\n"
                "11. Slope of the peak exercise ST segment (0-2)\n12. Number of major vessels colored by fluoroscopy (0-3)\n"
                "13. Thalassemia (1=normal, 2=fixed defect, 3=reversable defect)\n\n"
                "**Example:** 63,1,3,145,233,1,0,150,0,2.3,0,0,1"
            )
        }

    # 2. Check if user provided the 13 values (and we are expecting them)
    if state["prediction_mode"]:
        try:
            vector = [float(x.strip()) for x in latest_user_msg.split(",")]
            if len(vector) == 13:
                # Data received, reset prediction mode
                state["prediction_mode"] = False
                conversation_state[user_id] = state

                # Call the Scoring API
                try:
                    response = requests.post(
                        SCORING_API_URL,
                        json={"vector": vector},
                        timeout=60
                    )
                    response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
                    prediction_result = response.json().get("prediction") # e.g., 0 or 1

                    # Interpret result (adjust interpretation as needed based on your model)
                    if prediction_result == 1:
                        prediction_text = "potential risk detected"
                    elif prediction_result == 0:
                         prediction_text = "lower risk detected"
                    else:
                         prediction_text = "result unclear" # Handle unexpected results

                    return {
                        "type": "prediction_result",
                        "message": (f"Based on the provided data, the preliminary check indicates: **{prediction_text}**. \n\n"
                                    f"*Disclaimer: This is an AI-based estimation and not a medical diagnosis. "
                                    f"Please consult a qualified healthcare professional for any health concerns.*")
                    }
                except requests.exceptions.RequestException as e:
                    print(f"Error calling scoring API: {e}")
                    return {
                        "type": "error",
                        "message": "Sorry, I couldn't reach the prediction service right now. Please try again later."
                    }
                except Exception as e:
                    print(f"Error processing prediction: {e}")
                    return {
                        "type": "error",
                        "message": "Sorry, there was an error processing the prediction. Please check the format of your input."
                    }
            else:
                # Incorrect number of values
                return {
                    "type": "prompt_error",
                    "message": "It looks like you provided an incorrect number of values. Please provide exactly 13 comma-separated numbers."
                }
        except ValueError:
            # Input wasn't comma-separated numbers, assume it's regular chat
            # Reset prediction mode if they talk about something else
             state["prediction_mode"] = False
             conversation_state[user_id] = state
             pass # Fall through to regular chat
        except Exception as e:
             # Other parsing errors
             print(f"Parsing Error: {e}")
             return {
                 "type": "prompt_error",
                 "message": "I had trouble understanding those values. Please provide them as 13 comma-separated numbers (e.g., 63,1,3,...)."
             }

    # 3. Not prediction-related
    return None


# --- Chat Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Simple user identification (replace with actual auth/session ID)
    user_id = "default_user"

    messages = request.messages
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Extract latest user message
    latest_user_msg_obj = next((msg for msg in reversed(messages) if msg.role == "user"), None)
    if not latest_user_msg_obj:
         raise HTTPException(status_code=400, detail="No user message found")
    latest_user_msg = latest_user_msg_obj.content

    # --- Check for Prediction Flow ---
    prediction_response = handle_prediction_flow(user_id, latest_user_msg)

    assistant_response_content = ""
    if prediction_response:
        assistant_response_content = prediction_response["message"]
    else:
        # --- Regular Chat Flow with Memory ---
        # Reconstruct memory from history (simplified approach)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        for msg in messages[:-1]: # Load all but the latest user message into memory
            if msg.role == "user":
                memory.chat_memory.add_user_message(msg.content)
            elif msg.role == "assistant":
                memory.chat_memory.add_ai_message(msg.content)

        # Use LLMChain (or ConversationalRetrievalChain if using documents)
        # Note: Using a simple prompt, enhance as needed
        prompt = PromptTemplate(
            input_variables=["chat_history", "input"],
            template="""You are a helpful assistant. Respond to the user based on the conversation history.

Conversation History:
{chat_history}

User: {input}
Assistant:"""
        )
        chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True) # Set verbose=True for debugging

        # Get LLM response
        try:
            response = await chain.arun(latest_user_msg) # Use async run
            assistant_response_content = response
        except Exception as e:
            print(f"LLM Chain Error: {e}")
            assistant_response_content = "Sorry, I encountered an error while processing your request."

    # Append assistant response to history
    updated_history = [msg.dict() for msg in messages] + [{"role": "assistant", "content": assistant_response_content}]

    # Store updated state (optional, depends on state management needs)
    # conversation_state[user_id]['history'] = updated_history # Example

    return ChatResponse(chat_history=updated_history)

# --- Root Endpoint for Testing ---
@app.get("/")
def read_root():
    return {"message": "Heart Disease Assistant Backend is running"}