import requests

response = requests.post(
    "http://localhost:8000/process_content",
    json={
        "query": "Summarize this content",
        "file_path": "../docs/rag_model.png"
    }
)
print(response.json())