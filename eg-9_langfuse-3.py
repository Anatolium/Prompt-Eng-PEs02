# Убираем блок извлечения знаний
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
def conversation_cot_simple(user_input):
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

    При генерации карточки товара необходимо провести пошаговые рассуждения
    Логика рассуждений при генерации карточки товара:
    1) Определи Категорию товара (одно словo).
    2) Определи ключевую целевую аудиторию.
    3) 3 главные выгоды/УТП, которые нужно подчеркнуть.
    4) Список 4–6 характеристик из контекста, которые обязательно необходимо включить.
    5) Возможные пробелы в данных (если что-то важное не указано — коротко).


    Стиль: информативный, продающий, с упором на выгоду.
    Ограничение: до 500 слов.
    Проверь текст на отсутствие противоречий и повторов.

    Запрос пользователя:
    ''''
    {user_input}
    ''''

    Учти контекст о товаре при генерации карточки:
    '''
    {context}
    '''
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

    context = """
    Беспроводные наушники с активным шумоподавлением и встроенным микрофоном созданы специально для тренировок и активного образа жизни.
    Они обеспечивают чистое, мощное звучание даже в шумной обстановке благодаря технологии ANC.

    Эргономичная форма и мягкие амбушюры гарантируют комфортную посадку и устойчивость во время движений.
    Влагозащита по стандарту IPX5 защищает устройство от пота и дождя, что делает их идеальными для занятий спортом на улице или в зале.
    Встроенный микрофон с системой шумоподавления позволяет принимать звонки и использовать голосовые команды без прерывания тренировки.
    Поддержка Bluetooth 5.3 обеспечивает стабильное соединение без задержек. Наушники работают до 8 часов от одного заряда, а с зарядным кейсом — до 24 часов.
    Поддерживается быстрая зарядка: 10 минут = 1,5 часа воспроизведения. Совместимы с Android и iOS.

    Характеристики: беспроводное подключение Bluetooth 5.3; активное шумоподавление (ANC); встроенный микрофон; влагозащита IPX5; время автономной работы — до 8 часов (до 24 часов с кейсом);
    тип конструкции — внутриканальные; диапазон частот — 20 Гц–20 кГц; ёмкость аккумулятора — 40 мА·ч в наушнике, 400 мА·ч в кейсе; время зарядки — около 1,5 часа;
    интерфейс зарядки — USB Type-C; вес одного наушника — 4,5 г; цвет — чёрный.
    """

    # Создание цепочки через LCEL (вместо LLMChain)
    # Добавляем StrOutputParser, чтобы получать сразу строку, а не объект сообщения
    chain = prompt | llm | StrOutputParser()
    config = {"callbacks": [langfuse_handler]}
    response = chain.invoke({"user_input": user_input, "context": context}, config=config)
    # В LCEL response — это сразу строка (благодаря StrOutputParser)
    print(response)
    return response


if __name__ == "__main__":
    try:
        text = conversation_cot_simple('Наушники с шумоподавлением и микрофоном для тренировок')
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
