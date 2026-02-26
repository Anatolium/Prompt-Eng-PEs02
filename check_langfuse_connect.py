import os
import requests
from dotenv import load_dotenv

load_dotenv()

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")

# url = "https://us.cloud.langfuse.com"
# try:
#     response = requests.get(url, timeout=5)
#     print(f"✅ Доступ есть: {response.status_code}")
# except Exception as e:
#     print(f"❌ Нет доступа: {type(e).__name__}")

# Эндпоинт для отправки трассировки (API v1)
api_url = f"{LANGFUSE_BASE_URL}/api/public/ingestion"

print(f"🔍 Проверяем: {api_url}")

try:
    response = requests.post(
        api_url,
        auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
        json={"batch": []},  # Пустой батч для проверки
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    print(f"✅ API доступен: {response.status_code}")
    print(f"Ответ: {response.text[:200]}")
except Exception as e:
    print(f"❌ API недоступен: {type(e).__name__}: {e}")
