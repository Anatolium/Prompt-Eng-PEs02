# Prompt-Eng-PEs02
# 🧠 LLM Prompting & Memory Toolkit

Набор небольших Python-программ для работы с LLM: генерация обучающих материалов, refinement-цепочки, self-verification,
телеметрия и прямые API-вызовы.

Проекты демонстрируют разные паттерны взаимодействия с языковыми моделями — от простых запросов до многошаговых цепочек
с проверкой достоверности.

## 📦 Содержимое репозитория

### 🔹 pes02-dz.py — Refinement-цепочка для обучающих материалов

Интерактивная CLI-программа, которая:

- генерирует «набор для запоминания» по теме  
- затем автоматически улучшает результат (refinement)  
- усиливает конкретику и визуальные образы  
- добавляет чек-лист повторения  

**Ключевые особенности:**

- двухшаговая LLM-цепочка  
- разделение генерации и улучшения  
- параметризация креативности (`temperature`)  
- интерактивный цикл ввода  

**Использованные библиотеки:**

- `langchain-core`  
- `langchain-openai`  
- `python-dotenv`  
- `os`  

### 🔹 pes04-dz-OpenAI.py — LLM-гайд с self-verification и Langfuse

Программа строит структурированный гайд по техникам промптинга с внутренним циклом проверки.

**Реализованный пайплайн:**

1. Генерация материала  
2. Самопроверочные вопросы  
3. Независимая проверка  
4. Финальный синтез  

**Дополнительно:**

- трейсинг через Langfuse  
- подавление шумных логов OpenTelemetry  
- использование LCEL-цепочек  
- автоматический flush телеметрии  

**Использованные библиотеки:**

- `langchain-openai`  
- `langchain-core`  
- `langfuse`  
- `python-dotenv`  
- `logging`  
- `time`  
- `os`

### 🔹 pes04-dz-YandexGPT.py — Интеграция YandexGPT через OpenAI-совместимый API

Аналогичная self-verification цепочка, но с подключением модели YandexGPT через OpenAI-compatible endpoint.

**Что демонстрирует:**

- работу `ChatOpenAI` с альтернативным провайдером  
- настройку endpoint Yandex Cloud  
- устойчивые таймауты и retry  
- трейсинг в Langfuse  

**Особенности:**

- модель: `yandexgpt-lite`  
- увеличенный timeout  
- совместимость с новым OpenAI client  

**Использованные библиотеки:**

- `langchain-openai`  
- `langchain-core`  
- `langfuse`  
- `python-dotenv`  
- `logging`  
- `time`  
- `os`

### 🔹 pes05-dz.py — Прямой HTTP-запрос к Chat API

Минимальный пример работы с LLM без LangChain.

**Функциональность:**

- ручная сборка HTTP-запроса  
- отправка через `requests`  
- вывод сырого JSON  
- извлечение ответа модели  

**Что показывает:**

- низкоуровневую интеграцию  
- формат Chat Completions API  
- работу с headers и payload  

**Использованные библиотеки:**

- `requests`
- `json`  
- `python-dotenv`  
- `os`

## ⚙️ Основные паттерны, представленные в репозитории

- `Refinement chain`
- `Self-verification loop`
- `LCEL pipelines`
- `OpenAI-compatible endpoints`
- `LLM observability (Langfuse)`
- `Direct REST integration`
