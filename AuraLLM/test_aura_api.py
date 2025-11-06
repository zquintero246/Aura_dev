import requests
import time

API_URL = "http://127.0.0.1:8000/generate"

print("Conectado a AuraLM API")
print("Escribe 'salir' para terminar.\n")

history = []

while True:
    user_input = input("Tú: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ["salir", "exit", "quit"]:
        print("👋 Adiós!")
        break

    context = ""
    for msg in history[-4:]:
        context += f"{msg['role']}: {msg['content']}\n"
    context += f"User: {user_input}\nAssistant:"

    try:
        start = time.time()
        res = requests.post(API_URL, json={
            "prompt": context,
            "max_tokens": 256,
            "temperature": 0.8
        })
        latency = int((time.time() - start) * 1000)

        if res.status_code == 200:
            data = res.json()
            answer = data["output"].strip()
            print(f"Aura: {answer}\n⚡ ({latency} ms)\n")

            history.append({"role": "User", "content": user_input})
            history.append({"role": "Assistant", "content": answer})
        else:
            print("Error en la respuesta:", res.text)

    except KeyboardInterrupt:
        print("\nChat finalizado por el usuario.")
        break
    except Exception as e:
        print("Error:", e)
