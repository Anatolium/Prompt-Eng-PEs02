import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Конфигурация OpenAI ---
PROXYAPI_KEY = os.getenv("PROXYAPI_KEY")
if not PROXYAPI_KEY:
    raise ValueError("PROXYAPI_KEY не найден в переменных окружения")

BASE_URL = os.getenv("OPENAI_BASE_URL")
if not BASE_URL:
    raise ValueError("BASE_URL не найден в переменных окружения")

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=PROXYAPI_KEY,
    openai_api_base=BASE_URL
)

# # Создаем модель
# llm = ChatOpenAI(
#     model_name="gpt-4o-mini",
#     temperature=0.2,
#     openai_api_key="sk-proj-y"
# )

# Создаем промпт
prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
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

    """
)

chain = prompt | llm

print("Начнем диалог с моделью через LangChain! Введите 'exit' чтобы выйти.")

while True:
    # Наушники для спорта
    user_input = input("Вы: ")
    if user_input.lower() == "exit":
        print("Диалог завершен.")
        break

    response = chain.invoke({"user_input": user_input})
    print("\nМодель:\n", response.content, "\n")
