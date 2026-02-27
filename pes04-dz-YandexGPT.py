# Цепочка LLM для YandexGPT с трейсингом в Langfuse
import os
import logging
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI  # Работает с Yandex Cloud API!
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler
from langfuse import observe, get_client

# === Настройка логгирования (подавляем косметические ошибки OTLP) ===
logging.getLogger("opentelemetry.exporter.otlp.proto.http").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.sdk._shared_internal").setLevel(logging.CRITICAL)

load_dotenv()

# === Полезные настройки Langfuse ===
os.environ["LANGFUSE_TIMEOUT"] = "60"
os.environ["LANGFUSE_MAX_RETRIES"] = "3"
os.environ["LANGFUSE_FLUSH_AT"] = "1"
os.environ["LANGFUSE_FLUSH_INTERVAL"] = "1"
os.environ["LANGFUSE_DEBUG"] = "false"

# --- Конфигурация Yandex Cloud ---
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY")
YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER")

if not YANDEX_CLOUD_API_KEY:
    raise ValueError("YANDEX_CLOUD_API_KEY не найден в переменных окружения")
if not YANDEX_CLOUD_FOLDER:
    raise ValueError("YANDEX_CLOUD_FOLDER не найден в переменных окружения")

# --- Конфигурация Langfuse ---
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")

# Создаем обработчик обратных вызовов Langfuse
langfuse_handler = CallbackHandler()


@observe
def conversation(user_input):
    # Инициализация модели для YandexGPT через OpenAI-compatible endpoint
    llm = ChatOpenAI(
        model=f"gpt://{YANDEX_CLOUD_FOLDER}/yandexgpt-lite/latest",
        temperature=0.2,
        api_key=YANDEX_CLOUD_API_KEY,
        base_url="https://ai.api.cloud.yandex.net/v1",
        # ✅ Исправленные параметры совместимости:
        timeout=120,        # вместо request_timeout (новый стандарт OpenAI client)
        max_retries=3
    )

    # Создание шаблона промпта (без изменений)
    prompt = ChatPromptTemplate.from_template("""
    Ты — методист, который создаёт учебные материалы с встроенной проверкой достоверности.

    Задача: создать гайд по запоминанию техник промптинга Zero-Shot, Few-Shot, Chain-of-Thought, Chain-of-Verification,
    Chain-of-Note, Chain-of-Knowledge, используя цикл верификации.

    Следуй алгоритму:
    1. [Генерация] Создай первоначальный вариант: для каждой техники — название, суть в 5 словах, образ для запоминания.
    2. [Вопросы] Сформулируй 3–5 проверочных вопросов к своему же ответу, например: 
       • «Точно ли образ отражает суть техники?»
       • «Не перепутаны ли области применения?»
       • «Соответствует ли определение общепринятой практике?»
    3. [Проверка] Ответь на эти вопросы независимо, опираясь на стандартные определения промпт-инжиниринга.
    4. [Синтез] Исправь неточности и собери финальную версию, кратко пометив, что было уточнено.

    Формат финального ответа:
    • Техника | Суть | Образ-якорь | Проверенный триггер использования
    • В конце: раздел «Что я перепроверил и почему это важно» (3–4 пункта)

    Тон: внимательный дружелюбный редактор, который заботится о качестве.

    Запрос пользователя:
    '''
    {user_input}
    '''
    """)

    # Создание цепочки через LCEL
    chain = prompt | llm | StrOutputParser()
    config = {"callbacks": [langfuse_handler]}

    response = chain.invoke({"user_input": user_input}, config=config)

    print(response)
    return response


if __name__ == "__main__":
    # user_task = ("Объясни шесть техник промптинга: Zero-Shot, Few-Shot, Chain-of-Thought, Chain-of-Verification,"
    #              " Chain-of-Note и Chain-of-Knowledge, чтобы я мог их хорошо запомнить.")
    # user_task = ("Я считаю, что все эти «Chain-of-» техники — это лишь модный хайп. На практике они дают почти один "
    #              "и тот же результат, а остальное лишь усложняет. Особенно Chain-of-Verification и Chain-of-Knowledge"
    #              " кажутся мне лишними. Разубеди меня, если это не так.")
    user_task = ("Я готовлю мини-презентацию (10–12 минут) для своей учебной группы по техникам промптинга. Помоги мне"
                 " создать полный структурированный материал по техникам Zero-Shot, Few-Shot, Chain-of-Thought,"
                 "Chain-of-Verification, Chain-of-Note, Chain-of-Knowledge:"
                 "— с чёткими определениями и почему они работают,"
                 "— с хорошими мнемониками,"
                 "— с примерами из разных сфер (учёба, кодинг, анализ текстов),"
                 "— с таблицей сравнения (что легко перепутать и как различать),"
                 "— и с рекомендациями «когда какую технику брать»."
                 "Сделай так, чтобы мне было удобно сразу рассказывать по этому материалу.")

    try:
        text = conversation(f"{user_task}")
        print("\n✅ Основная логика завершена")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        raise
    finally:
        # Безопасное завершение Langfuse
        try:
            time.sleep(1)
            client = get_client()
            client.flush()
            print("✅ Langfuse: телеметрия отправлена")
        except Exception as e:
            print(f"⚠️ Langfuse: не удалось отправить телеметрию ({type(e).__name__})")
