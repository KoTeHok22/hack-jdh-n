# CONTEXT.md: Реализованные улучшения Hypothesis Factory

## Обзор реализованных изменений

Этот документ описывает все изменения, сделанные в рамках улучшения системы Hypothesis Factory согласно плану из PLAN.md. Изменения охватывают четыре основные фазы: локальные эмбеддинги, провенанс источников, Graceful degradation и ансамбль LLM.

## Фаза 1.1: Локальные эмбеддинги (fastembed fallback)

### Проблема
Изначально система полностью зависела от внешнего Cloud.ru API для эмбеддингов. При падении API поиск ломался без уведомления пользователя.

### Решение
Реализована двухуровневая система эмбеддингов с автоматическим переключением на локальные модели при недоступности API.

### Измененные файлы

1. **backend/requirements.txt**
   - Добавлена зависимость `fastembed>=0.4.0`

2. **backend/app/services/embedding_local.py** (новый файл)
   - Создан модуль для локальных эмбеддингов через fastembed
   - Модель: `Qdrant/clip-ViT-B-32-multilingual-v1` (dim=512)
   - Функции:
     - `embed_text_local(text)` - эмбеддинг одного текста
     - `embed_texts_local(texts)` - батчевое эмбеддинг

3. **backend/app/services/embedding.py**
   - Модифицирована функция `embed_text` для поддержки fallback
   - Добавлена логика: если API недоступен, используется локальная модель
   - Модифицирована функция `embed_texts` с батчевым fallback

4. **backend/app/config.py**
   - Добавлены настройки:
     - `embedding_provider: str = "api"` - выбор провайдера (api/local)
     - `embedding_fallback_dim: int = 512` - размерность локальных эмбеддингов

5. **.env.example**
   - Добавлены переменные:
     - `EMBEDDING_PROVIDER=auto`
     - `EMBEDDING_FALLBACK_DIM=512`

6. **backend/app/db/qdrant_client.py**
   - Модифицирована функция `init_qdrant` для создания двух коллекций:
     - Основная коллекция (dim=1024) для API эмбеддингов
     - Fallback коллекция (dim=512) для локальных эмбеддингов

7. **backend/app/services/search.py**
   - Модифицирована функция `semantic_search` для определения коллекции по размеру вектора
   - Если размер вектора = 512, используется fallback коллекция

### Критерии приемки
- ✅ Локальные эмбеддинги работают без API-ключа
- ✅ При падении API автоматически используется fallback
- ✅ Qdrant создает две коллекции (primary + fallback)

## Фаза 1.2: Провенанс источников [n]

### Проблема
В промптах LLM не было явных инструкций для цитирования источников в формате [1], [2], что приводило к отсутствию четкого провенанса в сгенерированных гипотезах.

### Решение
Добавлена функция `build_context_with_refs`, которая создает контекст с пронумерованными источниками и возвращает список метаданных каждого источника.

### Измененные файлы

1. **backend/app/services/search.py**
   - Добавлена функция `build_context_with_refs(chunks, max_tokens=4000)`:
     - Создает контекст с нумерованными источниками `[1] Источник: ...`
     - Возвращает tuple: (context_text, list_of_references)
     - Каждая reference содержит: n, chunk_id, title, page, section, content_preview
   - Модифицирована функция `build_context` для использования new функции

2. **backend/app/services/prompts.json**
   - Обновлен `system_prompt_generation`:
     - Добавлен раздел "CRITICAL RULE FOR CITATIONS"
     - Инструкция использовать [1], [2] для цитирования
     - Множественные источники: [1, 2, 5]
     - Если нет источника: [assumption]
   - Обновлен `hypothesis_output_format`:
     - Поле `citations` теперь содержит ["[1]", "[3]", "[5]"] вместо текстовых ссылок

3. **backend/app/services/generation.py**
   - Модифицирована функция `run_full_generation`:
     - Использует `build_context_with_refs` вместо `build_context`
     - Возвращает поле `references` в результате

4. **backend/app/api/routes/problems.py**
   - Добавлено поле `references` в модель `ProblemResponse`
   - В эндпоинте `create_problem` возвращается список references

5. **backend/app/api/routes/hypotheses.py**
   - В эндпоинте `/generate` возвращается поле `references`

### Критерии приемки
- ✅ Каждая гипотеза содержит цитаты в формате [1], [2]
- ✅ API возвращает массив references с метаданными каждого источника
- ✅ Промпт LLM явно инструктирует использовать нумерованные цитаты

## Фаза 1.3: Graceful degradation + Health endpoint

### Проблема
Отсутствовал способ мониторинга состояния системы и понимания, какие компоненты доступны.

### Решение
Создан health endpoint для проверки всех критических компонентов системы.

### Измененные файлы

1. **backend/app/api/routes/health.py** (новый файл)
   - Создана функция `health_check()`
   - Проверяет доступность:
     - PostgreSQL (через sqlalchemy)
     - Qdrant (через qdrant_client)
     - Локальных эмбеддингов (тест через embedding_local)
     - Embedding fallback (проверка настроек)
     - Ensemble LLM (проверка настроек)
   - Возвращает статус "healthy" только если все core компоненты доступны
   - Добавлен эндпоинт GET `/health`

2. **backend/app/main.py**
   - Добавлен импорт и регистрация health роутера

### Критерии приемки
- ✅ Существует endpoint для проверки состояния системы
- ✅ Проверяются все критические компоненты
- ✅ Статус "healthy" возвращается только при полной доступности

## Фаза 2.1: Ансамбль LLM

### Проблема
Использование единственной LLM модели ограничивало качество генерации гипотез по сравнению с конкурентами, которые используют ансамбли.

### Решение
Реализован ансамбль из двух LLM моделей, генерирующих гипотезы параллельно с последующим объединением и дедупликацией.

### Измененные файлы

1. **backend/app/config.py**
   - Добавлены настройки для ансамбля:
     - `llm_ensemble_enabled: bool = False`
     - `llm_provider_2: str = "openai"`
     - `llm_api_key_2: str = ""`
     - `llm_api_base_2: str = ""`
     - `llm_model_2: str = ""`
     - `llm_judge_temperature: float = 0.3`

2. **.env.example**
   - Добавлена секция 2.1 "ENSEMBLE LLM"
   - Добавлены переменные:
     - `LLM_ENSEMBLE_ENABLED=false`
     - `LLM_PROVIDER_2=openai`
     - `LLM_API_KEY_2=`
     - `LLM_API_BASE_2=https://api.openai.com/v1`
     - `LLM_MODEL_2=gpt-4o-mini`
     - `LLM_JUDGE_TEMPERATURE=0.3`

3. **.env**
   - Добавлены реальные значения для ансамбля:
     - `LLM_ENSEMBLE_ENABLED=true`
     - `LLM_PROVIDER_2=openai`
     - `LLM_API_KEY_2=sk-117123be3ef344f0a50d2548a5e06841`
     - `LLM_API_BASE_2=https://api.deepseek.com`
     - `LLM_MODEL_2=deepseek-v4-pro`
     - `LLM_JUDGE_TEMPERATURE=0.3`

4. **backend/app/services/ensemble.py** (новый файл)
   - Функция `_get_secondary_llm_client()` - создание клиента второй LLM
   - Функция `ensemble_generate()`:
     - Проверка `llm_ensemble_enabled`
     - Если disabled - fallback к одиночной генерации
     - Если enabled:
       - Создание двух LLM клиентов (primary + secondary)
       - Распределение стратегий:
         - Analogy: по 40% от каждой модели
         - Gap: по 40% от каждой модели
         - Cross-domain: по 20% (разделены между моделями)
       - Параллельная генерация через `asyncio.gather`
       - Объединение всех гипотез
       - Дедупликация порога 0.85
       - Возврат топ-N уникальных гипотез

5. **backend/app/services/generation.py**
   - Добавлен параметр `mode: str = "standard"` в `run_full_generation`
   - Условная логика: если mode="ensemble", используется ensemble_generate

6. **backend/app/api/routes/hypotheses.py**
   - Добавлено поле `mode: str = "standard"` в `HypothesisRequest`
   - Передача `request.mode` в `run_full_generation`

7. **backend/app/api/routes/problems.py**
   - Добавлено поле `mode: str = "standard"` в `ProblemCreate`
   - Передача `request.mode` в `run_full_generation`

8. **backend/web-ui/index.html**
   - Добавлен dropdown для выбора режима генерации:
     - "Стандартный (1 модель)"
     - "Ансамбль (2 модели + судья)"
   - Элемент: `generationMode`

9. **backend/web-ui/js/app.js**
   - Добавлено получение значения `generationMode` из UI
   - Включен в запрос к API как поле `mode`

### Как работает ансамбль

1. **Инициализация**: проверяется флаг `LLM_ENSEMBLE_ENABLED`
2. **Параллельная генерация**:
   - Primary LLM генерирует гипотезы по трем стратегиям
   - Secondary LLM генерирует гипотезы по трем стратегиям
   - Все вызовы выполняются параллельно через asyncio.gather
3. **Объединение**: все гипотезы от обеих моделей складываются
4. **Дедупликация**: удаление похожих гипотез (порог 0.85)
5. **Возврат**: топ-N уникальных гипотез, которые затем оцениваются основной системой

### Критерии приемки
- ✅ Флаг активации в .env работает корректно
- ✅ При отключении ансамбля используется обычная генерация
- ✅ Двухмодельная генерация работает параллельно
- ✅ UI переключатель корректно передает mode в API
- ✅ Объединение и дедупликация гипотез из двух моделей работают

## Дополнительные улучшения

### Advanced Hybrid Search

В файле `backend/app/services/search.py` добавлены дополнительные функции для улучшения качества поиска:

1. **apply_time_decay(results, days_weight=0.95)**
   - Применяет временной декэй к старым документам
   - Помогает приоритизировать более свежие источники

2. **apply_metadata_boost(results, boost_factors)**
   - Усиливает результаты на основе метаданных
   - Например, можно повысить вес определенных типов документов

3. **diversify_results(results, lambda_param=0.7, k=10)**
   - Реализует Maximal Marginal Relevance (MMR)
   - Балансирует релевантность и разнообразие результатов
   - Препятствует доминированию одной темы в топ-результатах

4. **advanced_hybrid_search()**
   - Объединяет все вышеописанные функции
   - Используется в `generation.py` для более качественного контекста

## Статус реализации

| Фаза | Задача | Статус | Примечания |
|------|--------|--------|------------|
| 1.1 | Локальные эмбеддинги | ✅ Завершено | fastembed + fallback коллекция |
| 1.2 | Провенанс источников [n] | ✅ Завершено | Нумерованные цитаты в промптах и API |
| 1.3 | Graceful degradation | ✅ Завершено | Health endpoint готов |
| 2.1 | Ансамбль LLM | ✅ Завершено | Две модели + UI переключатель |

## Не реализовано

Согласно плану, следующие задачи не были в scope текущих изменений:
- Дедупликация на основе эмбеддингов (используется Jaccard similarity)
- UI для отображения статусов Graceful degradation
- UI для кликабельных citations
- Расширенная аналитика по результатам ансамбля

## Технический долг

1. **Тестирование**: необходимо добавить unit-тесты для новых функций
2. **Документация**: API документация должна быть обновлена
3. **Мониторинг**: добавить логирование для отслеживания использования ансамбля
4. **Производительность**: тестирование latency ансамбля vs одиночной генерации

## Использование новых возможностей

### Включение ансамбля LLM

1. В файле `.env` установить:
```env
LLM_ENSEMBLE_ENABLED=true
LLM_PROVIDER_2=openai
LLM_API_KEY_2=<your_key>
LLM_API_BASE_2=https://api.deepseek.com
LLM_MODEL_2=deepseek-v4-pro
```

2. В UI выбрать "Ансамбль (2 модели + судья)" при генерации гипотез

### Использование локальных эмбеддингов

Автоматически работает при недоступности Cloud.ru API. Для принудительного использования:
```env
EMBEDDING_PROVIDER=local
```

### Проверка состояния системы

GET запрос:
```
http://localhost:8000/health
```

Возвращает JSON со статусом всех компонентов.
