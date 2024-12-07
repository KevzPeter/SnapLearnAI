import streamlit as st
import requests
import json
from typing import List, Dict
from datetime import datetime

# Initialize Streamlit app
st.title("📚 SnapLearn - Educational AI Assistant")
st.caption("Your offline learning companion powered by Local LLMs")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your educational assistant. How can I help you today?"}
    ]

# LM Studio API configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

def process_content(query: str, file_path: str = None) -> Dict:
    """Send query to LM Studio and get response"""
    try:
        # Define tools for content processing
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file_content",
                    "description": "Read and process educational content from files",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to be processed"
                            },
                            "operation": {
                                "type": "string",
                                "enum": ["summarize", "analyze", "extract"],
                                "description": "Operation to perform on content"
                            }
                        },
                        "required": ["file_path", "operation"]
                    }
                }
            }
        ]

        # Prepare the chat completion request
        payload = {
            "model": "lmstudio-community/qwen2.5-7b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an educational content processor. Process and analyze educational content effectively."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "tools": tools if file_path else None,
            "temperature": 0.7
        }

        response = requests.post(
            LM_STUDIO_URL,
            headers={"Content-Type": "application/json"},
            json=payload
        )

        if response.status_code == 200:
            response_data = response.json()
            
            # Check for tool calls
            if "choices" in response_data and response_data["choices"]:
                message = response_data["choices"][0]["message"]
                if "tool_calls" in message:
                    # Process tool calls and get results
                    tool_calls = message["tool_calls"]
                    results = []
                    
                    for tool_call in tool_calls:
                        function_args = json.loads(tool_call["function"]["arguments"])
                        # Execute the file processing function here
                        with open(function_args["file_path"], 'r') as f:
                            content = f.read()
                        results.append(content)
                    
                    # Get final response with results
                    final_prompt = f"Based on the content: {''.join(results)}, {query}"
                    final_payload = {
                        "model": "lmstudio-community/qwen2.5-7b-instruct",
                        "messages": [
                            {
                                "role": "user",
                                "content": final_prompt
                            }
                        ]
                    }
                    
                    final_response = requests.post(
                        LM_STUDIO_URL,
                        headers={"Content-Type": "application/json"},
                        json=final_payload
                    )
                    
                    return final_response.json()["choices"][0]["message"]
                else:
                    return message
            
            return {"content": "I couldn't process your request properly."}
        else:
            return {"content": f"Error: {response.status_code}"}
    except Exception as e:
        return {"content": f"Error: {str(e)}"}

# File uploader in sidebar
with st.sidebar:
    st.header("📁 Content Management")
    uploaded_file = st.file_uploader(
        "Upload educational content",
        type=['pdf', 'docx', 'txt', 'mp4']
    )
    
    if uploaded_file:
        st.success(f"File uploaded: {uploaded_file.name}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from LM Studio
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if uploaded_file:
                response = process_content(prompt, uploaded_file.name)
            else:
                response = process_content(prompt)
            
            response_content = response.get("content", "I apologize, but I couldn't process your request.")
            st.markdown(response_content)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_content})

# Add settings to sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your educational assistant. How can I help you today?"}
        ]
        st.rerun()