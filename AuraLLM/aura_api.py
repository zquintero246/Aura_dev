import requests
import json

url = "http://localhost:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "messages": [
       {"role": "system", "content": "Eres un asistente útil."},
       {"role": "user", "content": "¿Cuál es la capital de Colombia?"}
    ],
    "temperature": 0.7
}
resp = requests.post(url, headers=headers, data=json.dumps(data))
print(resp.json()["choices"][0]["message"]["content"])
