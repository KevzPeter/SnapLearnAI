import requests

response = requests.post(
    "http://localhost:8000/process_content",
    json={
        "query": "Summarize this content",
        "file_path": "../docs/demo_doc.pdf"
    }
)
print(response.json())