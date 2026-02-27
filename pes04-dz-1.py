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
    ''''
    {user_input}
    ''''

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
            # Фоновым потокам даём время завершить отправку
            time.sleep(1)
            client = get_client()
            client.flush()
            print("✅ Langfuse: телеметрия отправлена")
        except Exception as e:
            print(f"⚠️ Langfuse: не удалось отправить телеметрию ({type(e).__name__})")
