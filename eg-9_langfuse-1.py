# Цепочка LLM
import os
import logging
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler
from langfuse import observe, get_client

# === Настройка логгирования (подавляем косметические ошибки OTLP) ===
logging.getLogger("opentelemetry.exporter.otlp.proto.http").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.sdk._shared_internal").setLevel(logging.CRITICAL)

# Оставляем важные логи Langfuse
# logging.getLogger("langfuse").setLevel(logging.WARNING)

load_dotenv()

# === Полезные настройки Langfuse ===
os.environ["LANGFUSE_TIMEOUT"] = "60"
os.environ["LANGFUSE_MAX_RETRIES"] = "3"
os.environ["LANGFUSE_FLUSH_AT"] = "1"
os.environ["LANGFUSE_FLUSH_INTERVAL"] = "1"  # Сокращаем интервал до 1 секунды
os.environ["LANGFUSE_DEBUG"] = "false"

# --- Конфигурация OpenAI ---
PROXYAPI_KEY = os.getenv("PROXYAPI_KEY")
if not PROXYAPI_KEY:
    raise ValueError("PROXYAPI_KEY не найден в переменных окружения")

BASE_URL = os.getenv("OPENAI_BASE_URL")
if not BASE_URL:
    raise ValueError("BASE_URL не найден в переменных окружения")

# --- Конфигурация Langfuse ---
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")

# Создаем обработчик обратных вызовов Langfuse
langfuse_handler = CallbackHandler()


@observe
def conversation(user_input):
    # Инициализация модели (обновленные параметры для langchain-openai)
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # parameter name changed from model_name to model
        temperature=0.2,
        api_key=PROXYAPI_KEY,  # changed from openai_api_key
        base_url=BASE_URL  # changed from openai_api_base
    )

    # Создание шаблона промпта (используем ChatPromptTemplate)
    prompt = ChatPromptTemplate.from_template("""
    Роль модели: маркетолог и контент-редактор маркетплейсов (Ozon/Wildberries).
    Задача: создать карточку товара для маркетплейса по запросу пользователя.
    Карточка должна включать:
    -Название товара
    -Краткое описание (1–2 предложения)
    -Полное описание (5–7 предложений)
    -Преимущества (список)
    -Характеристики (таблица или список параметров)
    -SEO-ключевые слова (через запятую)

    Стиль: информативный, продающий, с упором на выгоду.
    Ограничение: до 500 слов.
    Проверь текст на отсутствие противоречий и повторов.

    Запрос пользователя:
    ''''
    {user_input}
    ''''

    Формат вывода:
    🔹 **Название товара:**
    [краткое и ёмкое название с УТП]

    🔹 **Краткое описание (1–2 предложения):**
    [одно сильное преимущество, мотивирующее купить]

    🔹 **Полное описание (5–7 предложений):**
    [описание функций, удобства, выгоды и сценария использования; ориентировано на покупателя]

    🔹 **Преимущества:**
    - [пункт 1]
    - [пункт 2]
    - [пункт 3]
    - [пункт 4]

    🔹 **Характеристики:**
    - [параметр: значение]
    - [параметр: значение]
    - [параметр: значение]
    - [параметр: значение]

    🔹 **SEO-ключевые слова:**
    [через запятую]
    """)

    # Создание цепочки через LCEL (вместо LLMChain)
    # Добавляем StrOutputParser, чтобы получать сразу строку, а не объект сообщения
    chain = prompt | llm | StrOutputParser()
    config = {"callbacks": [langfuse_handler]}
    response = chain.invoke({"user_input": user_input}, config=config)

    # В LCEL response — это сразу строка (благодаря StrOutputParser)
    print(response)
    return response


if __name__ == "__main__":
    try:
        text = conversation('Наушники с шумоподавлением и микрофоном для тренировок')
        print("\n✅ Основная логика завершена")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        raise
    finally:
        # Безопасное завершение Langfuse
        try:
            time.sleep(1)  # Даём фоновым потокам время завершить отправку
            client = get_client()
            client.flush()
            # langfuse_handler.flush()
            print("✅ Langfuse: телеметрия отправлена")
        except Exception as e:
            print(f"⚠️ Langfuse: не удалось отправить телеметрию ({type(e).__name__})")
