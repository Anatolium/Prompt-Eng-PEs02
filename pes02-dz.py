import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# --- Конфигурация ---
PROXYAPI_KEY = os.getenv("PROXYAPI_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

if not PROXYAPI_KEY or not BASE_URL:
    raise ValueError("Проверь переменные окружения PROXYAPI_KEY и OPENAI_BASE_URL")

TEMPERATURE = 0.6  # креативность для метафор

# --- Модель ---
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=TEMPERATURE,
    openai_api_key=PROXYAPI_KEY,
    openai_api_base=BASE_URL
)

# =========================
# ШАГ 1 — Создание черновика
# =========================

draft_prompt = PromptTemplate(
    input_variables=["user_topic"],
    template="""
Ты — эксперт по когнитивной науке и обучению взрослых.

Создай «Набор для запоминания» по теме:
"{user_topic}"

Для 5 ключевых техник или элементов укажи:
- Название
- Суть (до 10 слов)
- Визуальный образ
- Мини-метафору
- Практический якорь
- Ошибку-ловушку

Стиль: структурировано и понятно.
"""
)

draft_chain = draft_prompt | llm

# =========================
# ШАГ 2 — Улучшение (Refinement)
# =========================

refine_prompt = PromptTemplate(
    input_variables=["draft_output"],
    template="""
Ты — эксперт по когнитивной науке с фокусом на усиление запоминания.

Вот черновик обучающего материала:

--------------------
{draft_output}
--------------------

Твоя задача:
1. Сделать формулировки более конкретными.
2. Усилить визуальные образы (чтобы их можно было "увидеть").
3. Упростить суть до 7 слов максимум.
4. Сделать практические якоря более прикладными.
5. Убрать размытые и абстрактные фразы.

В конце добавь короткий чек-лист повторения (3–5 пунктов).

Стиль — уверенный, как у опытного преподавателя.
"""
)

refine_chain = refine_prompt | llm

# =========================
# Основной цикл
# =========================

print("Refinement-цепочка запущена.")
print("Введите тему для создания обучающего набора.")
print("Введите 'exit' для выхода.\n")

while True:
    user_input = input("Тема: ")

    if user_input.lower() == "exit":
        print("Работа завершена.")
        break

    # Шаг 1 — черновик
    draft_response = draft_chain.invoke({"user_topic": user_input})

    # Шаг 2 — улучшение
    final_response = refine_chain.invoke(
        {"draft_output": draft_response.content}
    )

    print(f"\nФинальный результат для Temperature={TEMPERATURE}:\n")
    print(final_response.content)
    print("\n" + "=" * 70 + "\n")
