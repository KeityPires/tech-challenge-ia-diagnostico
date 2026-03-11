import requests

url = "http://localhost:11434/api/generate"

data = {
    "model": "mistral",
    "prompt": "Explique brevemente o que é câncer de mama.",
    "stream": False
}

response = requests.post(url, json=data)

print(response.json()["response"])