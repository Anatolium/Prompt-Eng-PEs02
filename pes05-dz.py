import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
PROXYAPI_KEY = os.getenv("PROXYAPI_KEY")
if not PROXYAPI_KEY:
    raise ValueError("PROXYAPI_KEY не найден в переменных окружения")

BASE_URL = os.getenv("OPENAI_BASE_URL_CHAT")
if not BASE_URL:
    raise ValueError("BASE_URL не найден в переменных окружения")

headers = {
    "Authorization": f"Bearer {PROXYAPI_KEY}",  # Заголовок авторизации с Bearer токеном API ключа
    "Content-Type": "application/json"  # Указание, что тело запроса будет в формате JSON
}

data = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "Действуй как опытный и остроумный человек"},
        {"role": "user", "content": "Придумай 5 мотивационных цитат для программиста"}
    ],
    "temperature": 0.6
}

# Отправка POST-запроса к API с заголовками и сериализованными данными JSON
response = requests.post(BASE_URL, headers=headers, data=json.dumps(data))
print(response.status_code)
# Вывод JSON-ответа в удобном для чтения виде, с поддержкой русских символов
response_data = response.json()
print(json.dumps(response_data, indent=2, ensure_ascii=False))
print("\n=== ОТВЕТ МОДЕЛИ ===")
print(response_data["choices"][0]["message"]["content"])