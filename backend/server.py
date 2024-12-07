from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from typing import Optional
import json
import os

app = FastAPI()
LM_STUDIO_URL = "http://localhost:1234/v1"

class ContentRequest(BaseModel):
    query: str
    file_path: str

class ContentResponse(BaseModel):
    response: str
    file_processed: str

def process_content(file_path: str) -> str:
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
    return tools

def call_lm_studio(prompt: str, tools: list) -> dict:
    try:
        payload = {
            "model": "lmstudio-community/qwen2.5-7b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an educational content processor. Process and analyze educational content effectively."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "tools": tools,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{LM_STUDIO_URL}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/process_content")
async def process_content_endpoint(request: ContentRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    tools = process_content(request.file_path)
    prompt = f"Process the following request: {request.query} for file {request.file_path}"
    
    response = call_lm_studio(prompt, tools)
    
    if "choices" in response and response["choices"]:
        tool_calls = response["choices"][0]["message"].get("tool_calls", [])
        
        if tool_calls:
            # Process tool calls and get results
            results = []
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                
                # Execute the appropriate function based on tool call
                if function_name == "read_file_content":
                    # Implement actual file reading logic here
                    with open(arguments["file_path"], 'r') as f:
                        content = f.read()
                    results.append(content)
            
            # Get final response from LM Studio with results
            final_prompt = f"Based on the content: {''.join(results)}, {request.query}"
            final_response = call_lm_studio(final_prompt, [])
            
            return ContentResponse(
                response=final_response["choices"][0]["message"]["content"],
                file_processed=request.file_path
            )
    
    raise HTTPException(status_code=500, detail="Failed to process content")