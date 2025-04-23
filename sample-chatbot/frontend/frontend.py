# frontend/frontend.py
# (Keep your existing frontend code)
import streamlit as st
import requests
import os

st.set_page_config(page_title="Heart Disease Chatbot", page_icon="🫀")
st.title("🩺 Heart Disease Assistant")

st.markdown("### Chat with the assistant")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 How can I help you today?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
user_input = st.chat_input("Your message")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Get backend URL from environment variable, default if not set
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000/chat") # Default for local running

    try:
        response = requests.post(
            backend_url, # Use the variable here
            json={"messages": st.session_state.messages},
            timeout=60
        )
        response.raise_for_status() # Check for HTTP errors
        data = response.json()

        # Update history directly from backend response
        st.session_state.messages = data.get("chat_history", st.session_state.messages)

        # Display the last assistant message
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
             st.chat_message("assistant").write(st.session_state.messages[-1]["content"])
        else:
             # Handle case where backend might not have added an assistant message (error?)
             st.chat_message("assistant").write("Sorry, I couldn't get a response.")


    except requests.exceptions.RequestException as e:
        err_msg = f"⚠️ Error connecting to the backend: {e}"
        st.session_state.messages.append({"role": "assistant", "content": err_msg})
        st.chat_message("assistant").write(err_msg)
    except Exception as e:
        err_msg = f"⚠️ An unexpected error occurred: {e}"
        st.session_state.messages.append({"role": "assistant", "content": err_msg})
        st.chat_message("assistant").write(err_msg)