import os

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
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


# Задаем последовательность шагов рассуждения
class ProductCard(BaseModel):
    """Структура карточки товара для маркетплейса"""

    # Этап Chain-of-Knowledge
    marketing_practices: List[str] = Field(description="Типовые маркетинговые практики для данной категории")
    category_knowledge: List[str] = Field(description="Ключевые свойства категории и важные для покупки особенности")
    market_trends: List[str] = Field(description="Актуальные рыночные тренды и популярные формулировки")
    risks_and_mistakes: List[str] = Field(description="Типичные ошибки при создании карточек в этой категории")
    comparison_with_context: str = Field(description="Что совпадает и чего не хватает в контексте товара")

    # Этап Chain-of-Thought
    category: str = Field(description="Категория товара (одно слово)")
    target_audience: str = Field(description="Ключевая целевая аудитория")
    main_benefits: List[str] = Field(description="3 главные выгоды или УТП")
    required_characteristics: List[str] = Field(description="4–6 характеристик, которые нужно включить")
    data_gaps: str = Field(description="Недостающие данные или пробелы в контексте")

    # Итоговая карточка товара
    title: str = Field(description="Название товара")
    short_description: str = Field(description="Краткое описание (1–2 предложения)")
    full_description: str = Field(description="Полное описание (5–7 предложений)")
    advantages: List[str] = Field(description="Преимущества товара (список)")
    characteristics: List[str] = Field(description="Характеристики товара (список параметров)")
    seo_keywords: List[str] = Field(description="SEO-ключевые слова через запятую")


# Оборачиваем LLM так, что бы она следовала структуре выше
structured_llm = llm.with_structured_output(ProductCard)

query = 'Наушники с шумоподавлением и микрофоном для тренировок'
# print(structured_llm.invoke(query))
result = structured_llm.invoke(query)
print(result.model_dump())
