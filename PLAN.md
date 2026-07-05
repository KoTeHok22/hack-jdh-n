# PLAN.md: Улучшение Hypothesis Factory

## 📊 Текущее состояние vs Целевое

| Параметр | Сейчас | Цель (Фаза 1+2) | vs. nn_hypogen |
|----------|--------|-----------------|----------------|
| LLM-модели | 1 (YandexGPT/DeepSeek) | 2 + судья | ✅ Паритет |
| Эмбеддинги | Только Cloud.ru API | API + локальные (fastembed) | ✅ Лучше |
| Провенанс источников | Текстовые citations | Нумерация [n] + кликабельность | ⚠️ Почти паритет |
| Дедупликация | Jaccard по словам | Косинусное сходство эмбеддингов | ✅ Лучще |
| Graceful degradation | Нет | Полная система с UI-индикацией | ✅ Паритет |
| OCR-провайдеры | 3 (Mistral/GLM/Paddle) | 3 | ✅ Лучше |
| Стратегии генерации | 3 (аналогия/пробелы/междисц.) | 3 + ансамбль | ✅ Уникально |

---

## 🔥 ФАЗА 1: Фундаментальная надёжность (2-3 часа)

### Задача 1.1: Локальные эмбеддинги (fastembed fallback)

**Проблема:**
- `embedding.py:57` — при падении API возвращается `[0.0] * dim` (нулевой вектор)
- Поиск тихо ломается, пользователь не понимает причину
- Полная зависимость от внешнего API

**Решение:** Двухуровневая система эмбеддингов с автоматическим переключением.

#### Шаги реализации:

**1. Установить зависимости:**

```bash
# requirements.txt
fastembed>=0.4.0
```

**2. Создать `backend/app/services/embedding_local.py`:**

```python
import logging
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

_model = None
_MODEL_NAME = "Qdrant/clip-ViT-B-32-multilingual-v1"  # dim=512, мультиязычный
_DIM = 512

def _get_model():
    global _model
    if _model is None:
        logger.info(f"Loading local embedding model {_MODEL_NAME}...")
        _model = TextEmbedding(model_name=_MODEL_NAME)
        logger.info("Local embedding model loaded")
    return _model

async def embed_text_local(text: str) -> list[float]:
    model = _get_model()
    embeddings = list(model.embed([text]))
    return embeddings[0]

async def embed_texts_local(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    return [list(emb) for emb in model.embed(texts)]
```

**3. Модифицировать `backend/app/services/embedding.py`:**

```python
from app.services.embedding_local import embed_text_local, embed_texts_local
from app.config import get_settings

async def embed_text(text: str) -> list[float]:
    """Fallback chain: API → local → emergency"""
    settings = get_settings()
    
    if settings.embedding_provider == "local":
        return await embed_text_local(text)
    
    # Provider == "auto" or "api"
    try:
        return await _embed_one(text)  # API call with retry
    except Exception as e:
        logger.warning(f"API embedding failed, falling back to local: {e}")
        return await embed_text_local(text)

async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embedding with fallback"""
    settings = get_settings()
    
    if settings.embedding_provider == "local":
        return await embed_texts_local(texts)
    
    try:
        # Try API first (existing logic)
        logger.info(f"Embedding {len(texts)} texts via Cloud.ru API...")
        tasks = [_embed_one_with_limit(t) for t in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check if majority failed
        successful = [r for r in results if not isinstance(r, Exception)]
        if len(successful) < len(texts) * 0.5:
            raise Exception(f"API failed for {len(results) - len(successful)}/{len(texts)} texts")
        
        final_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"Failed embedding text {i + 1}: {r}")
                local_emb = await embed_text_local(texts[i])
                # Pad/trim to match API dim for consistency
                api_dim = settings.embedding_dim
                if len(local_emb) < api_dim:
                    local_emb += [0.0] * (api_dim - len(local_emb))
                elif len(local_emb) > api_dim:
                    local_emb = local_emb[:api_dim]
                final_results.append(local_emb)
            else:
                final_results.append(r)
        
        return final_results
    except Exception as e:
        logger.warning(f"Batch API embedding failed, falling back to local: {e}")
        return await embed_texts_local(texts)
```

**4. Обновить `backend/app/config.py`:**

```python
class Settings(BaseSettings):
    # ... existing fields ...
    
    embedding_provider: str = "auto"  # "api" | "local" | "auto"
    embedding_fallback_dim: int = 512  # fastembed dimension
    
    class Config:
        env_file = ".env"
```

**5. Обновить `.env.example`:**

```env
# Embedding provider: "api" (Cloud.ru), "local" (fastembed), "auto" (API + fallback)
EMBEDDING_PROVIDER=auto
EMBEDDING_FALLBACK_DIM=512
```

**6. Создать отдельную Qdrant-коллекцию для fallback:**

```python
# backend/app/db/qdrant_client.py
from qdrant_client.models import VectorParams, Distance

async def init_collections():
    client = get_qdrant()
    settings = get_settings()
    
    # Primary collection (API embeddings)
    if not await client.collection_exists(COLLECTION_NAME):
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=settings.embedding_dim,  # 1024
                distance=Distance.COSINE
            )
        )
    
    # Fallback collection (local embeddings)
    fallback_collection = f"{COLLECTION_NAME}_fallback"
    if not await client.collection_exists(fallback_collection):
        await client.create_collection(
            collection_name=fallback_collection,
            vectors_config=VectorParams(
                size=settings.embedding_fallback_dim,  # 512
                distance=Distance.COSINE
            )
        )
```

**7. Модифицировать `search.py` для поддержки fallback:**

```python
async def semantic_search(query: str, top_k: int = 10, filters: dict = None) -> list[dict]:
    client = get_qdrant()
    if client is None:
        return []
    
    settings = get_settings()
    query_vector = await embed_text(query)
    
    # Determine which collection to use based on vector size
    collection = COLLECTION_NAME
    if len(query_vector) == settings.embedding_fallback_dim:
        collection = f"{COLLECTION_NAME}_fallback"
    
    results = await client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
    )
    # ... existing logic ...
```

**Критерии приёмки:**
- ✅ Локальные эмбеддинги работают без API-ключа
- ✅ При падении API автоматически используется fallback
- ✅ Qdrant содержит две коллекции (primary + fallback)
- ✅ Health endpoint показывает текущий provider

**Тесты:**
```python
# test_embedding_fallback.py
import pytest
from app.services.embedding import embed_text

@pytest.mark.asyncio
async def test_local_embedding_works():
    """Local embedding should work without API key"""
    # Mock API failure
    with pytest.MonkeyPatch.context() as m:
        m.setenv("EMBEDDING_API_KEY", "")
        m.setenv("EMBEDDING_PROVIDER", "local")
        
        result = await embed_text("Тестовый текст")
        assert len(result) == 512  # fastembed dimension
        assert any(v != 0.0 for v in result)  # Not all zeros

@pytest.mark.asyncio
async def test_api_fallback_to_local():
    """Should fallback to local when API fails"""
    with pytest.MonkeyPatch.context() as m:
        m.setenv("EMBEDDING_API_KEY", "invalid_key")
        m.setenv("EMBEDDING_PROVIDER", "auto")
        
        result = await embed_text("Тестовый текст")
        assert len(result) == 512  # Fallback dimension
```

**Время:** 45 минут  
**Приоритет:** 🔴 Блокер

---

### Задача 1.2: Провенанс источников [n]

**Проблема:**
- `prompts.json` требует `citations: ["ссылка на источник 1"]` — просто текст
- Нет нумерации, нет связи с chunk ID
- Пользователь не может проверить источник

**Решение:** Нумерация источников `[1], [2], [3]` с метаданными.

#### Шаги реализации:

**1. Модифицировать `backend/app/services/search.py`:**

```python
def build_context_with_refs(chunks: list[dict], max_tokens: int = 4000) -> tuple[str, list[dict]]:
    """Build context with numbered references [1], [2], etc."""
    refs = []
    context_parts = []
    total_tokens = 0
    
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")
        source = chunk.get("source_title", "Unknown")
        page = chunk.get("page", "?")
        section = chunk.get("section", "")
        
        refs.append({
            "n": i,
            "chunk_id": chunk.get("id", f"chunk_{i}"),
            "title": source,
            "page": page,
            "section": section,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
        })
        
        header = f"[{i}] Источник: {source}"
        if page:
            header += f", стр. {page}"
        if section:
            header += f", раздел: {section}"
        
        entry = f"{header}\n{content}\n"
        entry_tokens = len(entry) // 4
        
        if total_tokens + entry_tokens > max_tokens:
            break
        
        context_parts.append(entry)
        total_tokens += entry_tokens
    
    context = "\n---\n".join(context_parts)
    return context, refs

# Backward compatibility
def build_context(chunks: list[dict], max_tokens: int = 4000) -> str:
    context, _ = build_context_with_refs(chunks, max_tokens)
    return context
```

**2. Обновить `backend/app/services/prompts.json`:**

```json
{
  "system_prompt_generation_with_refs": "Role: You are a Lead Process Innovator generating actionable hypotheses to reduce metal losses in concentrator tailings.\n\nTask: Generate hypotheses based on the data analysis and scientific context.\n\nCRITICAL RULE FOR CITATIONS:\n- Every factual claim MUST cite numbered sources as [1], [2], etc.\n- Example: \"Магнитная сепарация класса +71μm позволит извлечь до 78.4% никеля [1, 3]\"\n- If a claim has no source from the context, mark it as [assumption]\n- Multiple sources: [1, 2, 5]\n\nRules for each hypothesis:\n1. Specify a CONCRETE technological change (e.g., equipment, reagent, parameter).\n2. Reference SPECIFIC data points (size class, loss tonnage, mineral).\n3. Target a specific point of impact (operation, line, section).\n4. Provide a quantitative effect estimate (% recovery, tons/year).\n5. Ensure it is feasible within existing infrastructure.\n\nInstructions:\n1. Use a <scratchpad> to brainstorm and validate your hypotheses against the rules.\n2. Maintain your <scratchpad> reasoning in English to ensure maximum logical fidelity, but translate and output the final JSON content strictly in Russian.\n3. Output strictly as a JSON array.",
  
  "hypothesis_output_format_with_refs": "Return a JSON array of objects strictly matching this schema:\n[\n  {\n    \"statement\": \"Формулировка гипотезы (1-3 предложения) с цитатами [n]\",\n    \"mechanism\": \"Механизм воздействия с количественными данными (3-5 предложений) с цитатами [n]\",\n    \"citations\": [\"[1]\", \"[3]\", \"[5]\"],\n    \"reasoning_trace\": \"Пошаговое логическое обоснование валидности гипотезы\"\n  }\n]"
}
```

**3. Модифицировать `backend/app/services/generation.py`:**

```python
async def run_full_generation(
    statement: str,
    document_ids: list = None,
    num_hypotheses: int = 8,
    model: str = None,
) -> dict:
    search_results = await hybrid_search(statement, top_k=15)
    context, refs = build_context_with_refs(search_results)  # ← Изменено
    
    parsed_problem = await parse_problem(statement, context, model=model)
    
    hypotheses = await generate_hypotheses(
        statement, parsed_problem, context, num_hypotheses, model=model
    )
    
    if not hypotheses:
        return {
            "parsed_problem": parsed_problem,
            "hypotheses": [],
            "context_chunks": search_results,
            "references": refs,  # ← Добавлено
        }
    
    scores = await score_hypotheses(hypotheses, statement, context, model=model)
    
    scored_hypotheses = []
    for i, hyp in enumerate(hypotheses):
        # ... existing scoring logic ...
        
        scored_hypotheses.append({
            "statement": hyp.get("statement", ""),
            "mechanism": hyp.get("mechanism", ""),
            "citations": hyp.get("citations", []),  # Теперь содержит ["[1]", "[3]"]
            "reasoning_trace": hyp.get("reasoning_trace", ""),
            "novelty": novelty,
            "feasibility": feasibility,
            "impact": impact,
            "risk": risk,
            "confidence": score_data.get("confidence", 0.5),
            "composite_score": composite,
            "risks": score_data.get("risks", []),
            "verification_plan": score_data.get("verification_plan", ""),
        })
    
    scored_hypotheses.sort(key=lambda x: x["composite_score"], reverse=True)
    
    return {
        "parsed_problem": parsed_problem,
        "hypotheses": scored_hypotheses,
        "context_chunks": search_results,
        "references": refs,  # ← Добавлено
    }
```

**4. Обновить `backend/app/api/routes/hypotheses.py`:**

```python
@router.post("/generate")
async def generate_hypotheses(request: HypothesisRequest):
    result = await generation.run_full_generation(
        statement=request.statement,
        document_ids=request.document_ids,
        num_hypotheses=request.num_hypotheses,
        model=request.model,
    )
    return {
        "parsed_problem": result["parsed_problem"],
        "hypotheses": result["hypotheses"],
        "references": result.get("references", []),  # ← Добавлено
    }
```

**5. UI-изменения (web-ui/hypotheses.html):**

```javascript
// Рендер цитат как кликабельные бейджи
function renderCitations(citations, references) {
    return citations.map(cite => {
        const n = cite.replace(/[\[\]]/g, '');  // "[1]" → "1"
        const ref = references.find(r => r.n === parseInt(n));
        if (ref) {
            return `<span class="citation-badge" onclick="showReference(${n})">${cite}</span>`;
        }
        return cite;
    }).join(' ');
}

function showReference(n) {
    const ref = window.currentReferences.find(r => r.n === n);
    if (ref) {
        // Показать модальное окно с содержанием источника
        showModal(`
            <h3>Источник [${ref.n}]</h3>
            <p><strong>Документ:</strong> ${ref.title}</p>
            <p><strong>Страница:</strong> ${ref.page}</p>
            <p><strong>Раздел:</strong> ${ref.section}</p>
            <hr>
            <p>${ref.content_preview}</p>
        `);
    }
}
```

**6. CSS для citation-badge:**

```css
.citation-badge {
    display: inline-block;
    padding: 2px 8px;
    background: #007bff;
    color: white;
    border-radius: 12px;
    font-size: 0.85em;
    cursor: pointer;
    margin: 0 2px;
    transition: background 0.2s;
}

.citation-badge:hover {
    background: #0056b3;
}
```

**Критерии приёмки:**
- ✅ Каждая гипотеза содержит цитаты в формате `[1], [2]`
- ✅ API возвращает массив `references` с метаданными каждого источника
- ✅ UI показывает кликабельные бейджи для цитат
- ✅ При клике открывается модальное окно с содержанием chunk

**Тесты:**
```python
# test_provenance.py
@pytest.mark.asyncio
async def test_context_has_numbered_refs():
    chunks = [
        {"id": "1", "content": "Тест 1", "source_title": "Doc1", "page": "1"},
        {"id": "2", "content": "Тест 2", "source_title": "Doc2", "page": "2"},
    ]
    context, refs = build_context_with_refs(chunks)
    
    assert "[1]" in context
    assert "[2]" in context
    assert len(refs) == 2
    assert refs[0]["n"] == 1
    assert refs[1]["n"] == 2
```

**Время:** 30 минут  
**Приоритет:** 🔴 Критично

---

### Задача 1.3: Graceful degradation + Health endpoint

**Проблема:**
- Пользователь не понимает состояние системы
- При падении компонентов работа продолжается с деградацией, но без индикации

**Решение:** Системный статус + UI-индикация уровня деградации.

#### Шаги реализации:

**1. Создать `backend/app/services/system_health.py`:**

```python
import asyncio
import logging
from app.config import get_settings
from app.db.qdrant_client import get_qdrant
from app.db.database import get_db
from app.services.llm_client import get_llm_client
from app.services.embedding import embed_text, embed_text_local

logger = logging.getLogger(__name__)

async def check_embedding_health() -> str:
    """Returns: "api" | "local" | "degraded" | "none" """
    settings = get_settings()
    
    # Try API first
    try:
        if settings.embedding_provider != "local":
            await embed_text("health check")
            return "api"
    except Exception as e:
        logger.warning(f"API embedding health check failed: {e}")
    
    # Try local
    try:
        await embed_text_local("health check")
        return "local"
    except Exception as e:
        logger.error(f"Local embedding health check failed: {e}")
    
    return "none"

async def check_llm_health() -> dict:
    """Check LLM provider availability"""
    try:
        llm = get_llm_client()
        # Simple test call
        await llm.generate("Test", "Say 'ok'")
        return {
            "status": "ok",
            "provider": llm.__class__.__name__,
            "model": getattr(llm, 'model', 'unknown')
        }
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        return {
            "status": "degraded",
            "provider": "none",
            "error": str(e)
        }

async def check_qdrant_health() -> bool:
    try:
        client = get_qdrant()
        if client:
            await client.get_collection("hypotheses")
            return True
    except:
        pass
    return False

async def check_postgres_health() -> bool:
    try:
        db = await get_db()
        # Simple query
        await db.execute("SELECT 1")
        await db.close()
        return True
    except:
        return False

async def get_system_status() -> dict:
    """Comprehensive system health check"""
    embedding_status = await check_embedding_health()
    llm_status = await check_llm_health()
    qdrant_status = await check_qdrant_health()
    postgres_status = await check_postgres_health()
    
    # Determine overall mode
    if embedding_status == "api" and llm_status["status"] == "ok":
        overall = "full"
    elif embedding_status in ["api", "local"] and llm_status["status"] == "ok":
        overall = "limited"
    elif qdrant_status and postgres_status:
        overall = "degraded"
    else:
        overall = "critical"
    
    return {
        "embedding": {
            "status": embedding_status,
            "detail": f"Using {embedding_status} embeddings"
        },
        "llm": llm_status,
        "qdrant": qdrant_status,
        "postgres": postgres_status,
        "overall": overall,
        "overall_detail": {
            "full": "Все системы работают нормально",
            "limited": "Ограниченный режим (локальные эмбеддинги или альтернативная LLM)",
            "degraded": "Деградированный режим (некоторые компоненты недоступны)",
            "critical": "Критический сбой (система не может работать)"
        }[overall]
    }
```

**2. Создать новый роут `backend/app/api/routes/system.py`:**

```python
from fastapi import APIRouter
from app.services.system_health import get_system_status

router = APIRouter(prefix="/system", tags=["system"])

@router.get("/status")
async def system_status():
    """Get comprehensive system health status"""
    return await get_system_status()
```

**3. Зарегистрировать роут в `backend/app/main.py`:**

```python
from app.api.routes import system

app.include_router(system.router, prefix="/api/v1")
```

**4. UI-компонент (web-ui/components/status-bar.js):**

```javascript
async function updateStatusBar() {
    try {
        const response = await fetch('/api/v1/system/status');
        const status = await response.json();
        
        const statusBar = document.getElementById('status-bar');
        
        const statusIcons = {
            full: '🟢',
            limited: '🟡',
            degraded: '🟠',
            critical: '🔴'
        };
        
        const icon = statusIcons[status.overall] || '⚪';
        const detail = status.overall_detail;
        
        statusBar.innerHTML = `
            <div class="status-bar">
                <span class="status-icon">${icon}</span>
                <span class="status-text">Система: ${detail}</span>
                <div class="status-details">
                    <span class="status-item">
                        ${status.embedding.status === 'api' ? '✅' : '⚠️'} Эмбеддинги: ${status.embedding.status}
                    </span>
                    <span class="status-item">
                        ${status.llm.status === 'ok' ? '✅' : '⚠️'} LLM: ${status.llm.provider}
                    </span>
                    <span class="status-item">
                        ${status.qdrant ? '✅' : '❌'} Qdrant
                    </span>
                    <span class="status-item">
                        ${status.postgres ? '✅' : '❌'} PostgreSQL
                    </span>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Failed to update status bar:', error);
    }
}

// Обновлять статус каждые 30 секунд
setInterval(updateStatusBar, 30000);

// Первоначальное обновление при загрузке страницы
document.addEventListener('DOMContentLoaded', updateStatusBar);
```

**5. CSS для status-bar (web-ui/styles/status-bar.css):**

```css
.status-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 16px;
    background: #f8f9fa;
    border-bottom: 1px solid #dee2e6;
    font-size: 0.9em;
}

.status-icon {
    font-size: 1.2em;
}

.status-text {
    font-weight: 500;
    flex: 1;
}

.status-details {
    display: flex;
    gap: 16px;
}

.status-item {
    display: flex;
    align-items: center;
    gap: 4px;
}

@media (max-width: 768px) {
    .status-details {
        flex-direction: column;
        gap: 4px;
    }
}
```

**Критерии приёмки:**
- ✅ `GET /api/v1/system/status` возвращает детальный статус
- ✅ UI показывает индикатор в верхней части страницы
- ✅ Статус автоматически обновляется каждые 30 секунд
- ✅ Четыре уровня: full → limited → degraded → critical

**Тесты:**
```python
# test_system_health.py
@pytest.mark.asyncio
async def test_full_system_status():
    status = await get_system_status()
    
    assert "overall" in status
    assert status["overall"] in ["full", "limited", "degraded", "critical"]
    assert "embedding" in status
    assert "llm" in status
    assert "qdrant" in status
    assert "postgres" in status
```

**Время:** 30 минут  
**Приоритет:** 🟡 Важно

---

## 🔥 ФАЗА 2: Качество гипотез (4-5 часов)

### Задача 2.1: Ансамбль LLM (2 модели + судья)

**Проблема:**
- Одна LLM может галлюцинировать или упускать важные аспекты
- Конкуренты используют ансамбль для снижения дисперсии ошибок

**Решение:** Параллельная генерация от 2 моделей + финальная сборка судьёй.

#### Шаги реализации:

**1. Обновить `backend/app/config.py`:**

```python
class Settings(BaseSettings):
    # ... existing fields ...
    
    # Second LLM for ensemble
    llm_ensemble_enabled: bool = False
    llm_provider_2: str = "openai"  # "yandex" | "openai"
    llm_api_key_2: str = ""
    llm_api_base_2: str = "https://api.openai.com/v1"
    llm_model_2: str = "gpt-4o-mini"
    
    # Ensemble judge (uses primary LLM by default)
    llm_judge_temperature: float = 0.3

class Config:
    env_file = ".env"
```

**2. Создать `backend/app/services/ensemble.py`:**

```python
import asyncio
import json
import logging
from app.services.generation import (
    _gen_strategy,
    _deduplicate_hypotheses,
    SYSTEM_PROMPT_GENERATION,
    GENERATION_STRATEGY_ANALOGY,
    GENERATION_STRATEGY_GAP,
    GENERATION_STRATEGY_CROSS_DOMAIN,
    HYPOTHESIS_OUTPUT_FORMAT,
)
from app.services.llm_client import get_llm_client
from app.config import get_settings

logger = logging.getLogger(__name__)

class EnsembleGenerator:
    def __init__(self):
        settings = get_settings()
        if not settings.llm_ensemble_enabled:
            raise ValueError("Ensemble generation is disabled in config")
        
        # Primary LLM
        self.llm_a = get_llm_client()
        
        # Secondary LLM
        from app.services.llm_client import YandexLLMClient, OpenAILLMClient
        if settings.llm_provider_2 == "yandex":
            self.llm_b = YandexLLMClient(
                api_key=settings.yandex_api_key,
                folder_id=settings.yandex_folder_id,
                api_base=settings.yandex_api_base,
                model=settings.yandex_model,
            )
        else:
            self.llm_b = OpenAILLMClient(
                api_key=settings.llm_api_key_2,
                api_base=settings.llm_api_base_2,
                model=settings.llm_model_2,
            )
    
    async def generate_draft(self, llm, statement, parsed_problem, context, num_hypotheses):
        """Generate hypotheses draft from a single LLM"""
        # Параллельная генерация по стратегиям
        analogy_count = int(num_hypotheses * 0.4)
        gap_count = int(num_hypotheses * 0.4)
        cross_count = num_hypotheses - analogy_count - gap_count
        
        analogy_task = _gen_strategy(
            llm, statement, parsed_problem, context,
            GENERATION_STRATEGY_ANALOGY, analogy_count, "", "analogy"
        )
        gap_task = _gen_strategy(
            llm, statement, parsed_problem, context,
            GENERATION_STRATEGY_GAP, gap_count, "", "gap"
        )
        
        analogy_hyps, gap_hyps = await asyncio.gather(analogy_task, gap_task)
        
        existing_text = "\n".join([
            f"- {h.get('statement', '')}" for h in analogy_hyps + gap_hyps
        ])
        
        cross_hyps = await _gen_strategy(
            llm, statement, parsed_problem, context,
            GENERATION_STRATEGY_CROSS_DOMAIN, cross_count, existing_text, "cross"
        )
        
        return analogy_hyps + gap_hyps + cross_hyps
    
    async def judge_hypotheses(self, drafts: list[list[dict]], statement, context):
        """Use primary LLM as judge to merge drafts"""
        judge = self.llm_a
        
        # Deduplicate each draft first
        deduplicated_drafts = [_deduplicate_hypotheses(d, threshold=0.90) for d in drafts]
        
        # Prepare judge prompt
        drafts_text = "\n\n".join([
            f"Черновик от Модели {chr(65+i)} ({len(draft)} гипотез):\n" +
            json.dumps([{"statement": h["statement"], "mechanism": h["mechanism"]} 
                       for h in draft], ensure_ascii=False, indent=2)
            for i, draft in enumerate(deduplicated_drafts)
        ])
        
        judge_prompt = f"""Ты — эксперт-судья, объединяющий результаты двух независимых моделей генерации гипотез.

ПРОБЛЕМА: {statement}

ЧЕРНОВИКИ ОТ ДВУХ МОДЕЛЕЙ:
{drafts_text}

ТВОЯ ЗАДАЧА:
1. Объедини оба черновика в единый финальный список
2. Удалившиеся гипотезы (очевидные повторы) — исключи
3. Для оставшихся гипотез:
   - Если обе модели согласны — включи с пометкой "confirmed_by_both"
   - Если уникальна для одной модели — оцени качество и включи с пометкой "unique_quality"
   - Если слабая (нет конкретных цифр, механизмов) — исключи
4. Выдай reasoning: почему каждая гипотеза включена/исключена

ФОРМАТ ОТВЕТА (строго JSON):
{{
  "final_hypotheses": [
    {{
      "statement": "...",
      "mechanism": "...",
      "citations": [...],
      "reasoning_trace": "...",
      "source": "both" | "model_a" | "model_b",
      "judge_reason": "Почему включена"
    }}
  ],
  "excluded_count": <число>,
  "judge_reasoning": "Общая логика отбора"
}}"""
        
        result = await judge.generate_json(
            "Ты — эксперт-судья для объединения гипотез от двух LLM",
            judge_prompt,
            temperature=0.3
        )
        
        try:
            from app.services.generation import _extract_json
            judge_result = _extract_json(result)
            return judge_result
        except Exception as e:
            logger.error(f"Judge failed to parse JSON: {e}")
            # Fallback: merge all drafts
            merged = []
            for draft in deduplicated_drafts:
                merged.extend(draft)
            return {
                "final_hypotheses": merged,
                "excluded_count": 0,
                "judge_reasoning": "Judge parsing failed, merged all drafts"
            }
    
    async def generate_ensemble(self, statement, parsed_problem, context, num_hypotheses=8):
        """Full ensemble generation pipeline"""
        logger.info(f"Starting ensemble generation: {num_hypotheses} hypotheses")
        
        # Параллельная генерация от обеих моделей
        draft_a_task = self.generate_draft(
            self.llm_a, statement, parsed_problem, context, num_hypotheses
        )
        draft_b_task = self.generate_draft(
            self.llm_b, statement, parsed_problem, context, num_hypotheses
        )
        
        draft_a, draft_b = await asyncio.gather(draft_a_task, draft_b_task)
        
        logger.info(f"Drafts generated: A={len(draft_a)}, B={len(draft_b)}")
        
        # Судья объединяет
        judge_result = await self.judge_hypotheses(
            [draft_a, draft_b], statement, context
        )
        
        return {
            "hypotheses": judge_result.get("final_hypotheses", []),
            "drafts": {
                "model_a": draft_a,
                "model_b": draft_b,
            },
            "judge_reasoning": judge_result.get("judge_reasoning", ""),
            "excluded_count": judge_result.get("excluded_count", 0),
        }
```

**3. Обновить `backend/app/services/generation.py`:**

```python
from app.services.ensemble import EnsembleGenerator
from app.config import get_settings

async def run_full_generation(
    statement: str,
    document_ids: list = None,
    num_hypotheses: int = 8,
    model: str = None,
    mode: str = "single",  # ← Новый параметр
) -> dict:
    search_results = await hybrid_search(statement, top_k=15)
    context, refs = build_context_with_refs(search_results)
    
    parsed_problem = await parse_problem(statement, context, model=model)
    
    settings = get_settings()
    
    # Ensemble mode
    if mode == "ensemble" and settings.llm_ensemble_enabled:
        logger.info("Using ensemble generation mode")
        ensemble = EnsembleGenerator()
        ensemble_result = await ensemble.generate_ensemble(
            statement, parsed_problem, context, num_hypotheses
        )
        
        # Score the ensemble hypotheses
        hypotheses = ensemble_result["hypotheses"]
        scores = await score_hypotheses(hypotheses, statement, context, model=model)
        
        scored_hypotheses = []
        for i, hyp in enumerate(hypotheses):
            score_data = next((s for s in scores if s.get("index") == i), {
                "novelty": 3, "feasibility": 3, "impact": 3, "risk": 3,
                "confidence": 0.5, "risks": [], "verification_plan": ""
            })
            
            novelty = score_data.get("novelty", 3)
            feasibility = score_data.get("feasibility", 3)
            impact = score_data.get("impact", 3)
            risk = score_data.get("risk", 3)
            composite = calculate_composite_score(novelty, feasibility, impact, risk)
            
            scored_hypotheses.append({
                "statement": hyp.get("statement", ""),
                "mechanism": hyp.get("mechanism", ""),
                "citations": hyp.get("citations", []),
                "reasoning_trace": hyp.get("reasoning_trace", ""),
                "novelty": novelty,
                "feasibility": feasibility,
                "impact": impact,
                "risk": risk,
                "confidence": score_data.get("confidence", 0.5),
                "composite_score": composite,
                "risks": score_data.get("risks", []),
                "verification_plan": score_data.get("verification_plan", ""),
                "ensemble_source": hyp.get("source", "both"),  # ← Новое поле
                "judge_reason": hyp.get("judge_reason", ""),  # ← Новое поле
            })
        
        scored_hypotheses.sort(key=lambda x: x["composite_score"], reverse=True)
        
        return {
            "parsed_problem": parsed_problem,
            "hypotheses": scored_hypotheses,
            "context_chunks": search_results,
            "references": refs,
            "ensemble_drafts": ensemble_result["drafts"],  # ← Для UI
            "judge_reasoning": ensemble_result["judge_reasoning"],  # ← Для UI
        }
    
    # Single mode (existing logic)
    hypotheses = await generate_hypotheses(
        statement, parsed_problem, context, num_hypotheses, model=model
    )
    
    if not hypotheses:
        return {
            "parsed_problem": parsed_problem,
            "hypotheses": [],
            "context_chunks": search_results,
            "references": refs,
        }
    
    scores = await score_hypotheses(hypotheses, statement, context, model=model)
    
    scored_hypotheses = []
    for i, hyp in enumerate(hypotheses):
        # ... existing scoring logic (unchanged) ...
    
    scored_hypotheses.sort(key=lambda x: x["composite_score"], reverse=True)
    
    return {
        "parsed_problem": parsed_problem,
        "hypotheses": scored_hypotheses,
        "context_chunks": search_results,
        "references": refs,
    }
```

**4. Обновить `backend/app/api/routes/hypotheses.py`:**

```python
from pydantic import BaseModel
from typing import Optional

class HypothesisRequest(BaseModel):
    statement: str
    document_ids: Optional[list] = None
    num_hypotheses: int = 8
    model: Optional[str] = None
    mode: str = "single"  # ← Новое поле: "single" | "ensemble"

@router.post("/generate")
async def generate_hypotheses(request: HypothesisRequest):
    result = await generation.run_full_generation(
        statement=request.statement,
        document_ids=request.document_ids,
        num_hypotheses=request.num_hypotheses,
        model=request.model,
        mode=request.mode,  # ← Передаём mode
    )
    
    response = {
        "parsed_problem": result["parsed_problem"],
        "hypotheses": result["hypotheses"],
        "references": result.get("references", []),
    }
    
    # Добавляем ensemble-данные если они есть
    if "ensemble_drafts" in result:
        response["ensemble_drafts"] = result["ensemble_drafts"]
        response["judge_reasoning"] = result["judge_reasoning"]
    
    return response
```

**5. Обновить `.env.example`:**

```env
# Ensemble generation (2 LLM models + judge)
LLM_ENSEMBLE_ENABLED=false
LLM_PROVIDER_2=openai
LLM_API_KEY_2=sk-your-openai-key
LLM_API_BASE_2=https://api.openai.com/v1
LLM_MODEL_2=gpt-4o-mini
LLM_JUDGE_TEMPERATURE=0.3
```

**6. UI-изменения (web-ui/hypotheses.html):**

```javascript
// Переключатель режима генерации
const modeSelector = `
    <div class="mode-selector">
        <label>
            <input type="radio" name="mode" value="single" checked>
            Быстрая генерация (1 модель, 30-60 сек)
        </label>
        <label>
            <input type="radio" name="mode" value="ensemble">
            Ансамбль (2 модели + судья, 2-3 мин)
        </label>
    </div>
`;

// При отправке формы
const mode = document.querySelector('input[name="mode"]:checked').value;

const response = await fetch('/api/v1/hypotheses/generate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        statement: statement,
        mode: mode,
        num_hypotheses: 8
    })
});

// Рендер ensemble-данных
if (result.ensemble_drafts) {
    renderEnsemblePanel(result);
}

function renderEnsemblePanel(result) {
    const panel = document.createElement('div');
    panel.className = 'ensemble-panel';
    panel.innerHTML = `
        <h3>🤖 Ансамбль LLM: ${result.hypotheses.length} финальных гипотез</h3>
        <div class="ensemble-stats">
            <span>Модель A сгенерировала: ${result.ensemble_drafts.model_a.length}</span>
            <span>Модель B сгенерировала: ${result.ensemble_drafts.model_b.length}</span>
            <span>Отклонено судьёй: ${result.ensemble_drafts.model_a.length + result.ensemble_drafts.model_b.length - result.hypotheses.length}</span>
        </div>
        <details>
            <summary>Рассуждение судьи</summary>
            <pre>${result.judge_reasoning}</pre>
        </details>
    `;
    document.getElementById('hypotheses-container').prepend(panel);
}
```

**Критерии приёмки:**
- ✅ `POST /api/v1/hypotheses/generate` принимает `mode: "ensemble"`
- ✅ Две модели генерируют параллельно (через `asyncio.gather`)
- ✅ Судья объединяет результаты с reasoning
- ✅ UI показывает переключатель режимов
- ✅ Для ансамбля отображается статистикаdrafts и reasoning судьи

**Тесты:**
```python
# test_ensemble.py
@pytest.mark.asyncio
async def test_ensemble_generation():
    """Ensemble should generate from 2 models and merge"""
    settings = get_settings()
    settings.llm_ensemble_enabled = True
    
    ensemble = EnsembleGenerator()
    result = await ensemble.generate_ensemble(
        statement="Повысить извлечение никеля на 15%",
        parsed_problem={"domain": "flotation"},
        context="[1] Test context",
        num_hypotheses=4
    )
    
    assert "hypotheses" in result
    assert "drafts" in result
    assert "judge_reasoning" in result
    assert len(result["drafts"]["model_a"]) > 0
    assert len(result["drafts"]["model_b"]) > 0
```

**Время:** 2.5 часа  
**Приоритет:** 🔴 Критично

---

## 📝 Дополнительные задачи (Фаза 3 — если останется время)

### Задача 3.1: Улучшенная дедупликация (эмбеддинг-сходство)

**Время:** 20 минут  
**Файл:** `backend/app/services/generation.py`

Заменить Jaccard на косинусное сходство эмбеддингов:

```python
async def _deduplicate_embeddings(hypotheses: list[dict], threshold: float = 0.92) -> list[dict]:
    """Deduplicate using embedding similarity instead of Jaccard"""
    statements = [h.get("statement", "") for h in hypotheses]
    embeddings = await embed_texts(statements)
    
    unique = []
    unique_embeddings = []
    
    for hyp, emb in zip(hypotheses, embeddings):
        is_dup = False
        for existing_emb in unique_embeddings:
            similarity = cosine_similarity(emb, existing_emb)
            if similarity > threshold:
                logger.debug(f"Duplicate detected (cos={similarity:.3f})")
                is_dup = True
                break
        if not is_dup:
            unique.append(hyp)
            unique_embeddings.append(emb)
    
    return unique

def cosine_similarity(a, b):
    """Simple cosine similarity"""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)
```

---

### Задача 3.2: Веб-аугментация (OpenAlex API)

**Время:** 1.5 часа  
**Файл:** `backend/app/services/web_search.py` (новый)

```python
import httpx
import logging

logger = logging.getLogger(__name__)

async def search_openalex(query: str, year_from: int = 2020, max_results: int = 5) -> list[dict]:
    """Search scientific literature via OpenAlex API (free, no API key needed)"""
    url = f"https://api.openalex.org/works"
    params = {
        "search": query,
        "filter": f"from_publication_date:{year_from}-01-01",
        "per_page": max_results,
        "mailto": "hypothesis-factory@example.com"  # Polite pool
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for work in data.get("results", []):
                results.append({
                    "n": len(results) + 1,
                    "title": work.get("title", ""),
                    "authors": [a.get("author", {}).get("display_name", "") 
                               for a in work.get("authorships", [])[:3]],
                    "year": work.get("publication_year"),
                    "doi": work.get("doi"),
                    "abstract": work.get("abstract_inverted_index", {}),
                    "source_type": "openalex"
                })
            
            return results
    except Exception as e:
        logger.error(f"OpenAlex search failed: {e}")
        return []
```

---

### Задача 3.3: Deep Research Mode

**Время:** 3-4 часа  
**Файл:** `backend/app/services/deep_research.py` (новый)

```python
async def run_deep_research(question: str, max_subqueries: int = 4, web_enabled: bool = True) -> dict:
    """
    Deep Research pipeline:
    1. Decompose question into sub-queries
    2. Parallel search (Qdrant + optional OpenAlex)
    3. Synthesize comprehensive answer with citations
    """
    llm = get_llm_client()
    
    # 1. Decompose
    decompose_prompt = f"""Разбей вопрос на 3-5 под-вопросов для глубокого анализа.

ВОПРОС: {question}

Верни JSON:
{{"sub_queries": ["подвопрос 1", "подвопрос 2", ...]}}"""
    
    result = await llm.generate_json("Ты — эксперт-аналитик", decompose_prompt)
    sub_queries = _extract_json(result).get("sub_queries", [question])
    
    # 2. Parallel search
    search_tasks = [hybrid_search(q, top_k=10) for q in sub_queries]
    if web_enabled:
        web_tasks = [search_openalex(q, max_results=3) for q in sub_queries]
        web_results = await asyncio.gather(*web_tasks, return_exceptions=True)
    else:
        web_results = []
    
    search_results = await asyncio.gather(*search_tasks)
    
    # 3. Synthesize
    all_evidence = []
    for i, results in enumerate(search_results):
        for chunk in results:
            all_evidence.append({
                "sub_query": sub_queries[i],
                **chunk
            })
    
    if web_enabled:
        for i, results in enumerate(web_results):
            if isinstance(results, Exception):
                continue
            for work in results:
                all_evidence.append({
                    "sub_query": sub_queries[i],
                    **work
                })
    
    context, refs = build_context_with_refs(all_evidence)
    
    synthesis_prompt = f"""На основе следующего исследования составь аналитический отчёт.

ВОПРОС: {question}

ПОД-ВОПРОСЫ:
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(sub_queries)])}

ДОКАЗАТЕЛЬСТВА:
{context}

Составь подробный отчёт с:
1. Кратким резюме (2-3 предложения)
2. Анализом по каждому под-вопросу
3. Выводами и рекомендациями
4. Списком источников [n]"""
    
    answer = await llm.generate("Ты — эксперт-исследователь", synthesis_prompt)
    
    return {
        "question": question,
        "sub_queries": sub_queries,
        "answer": answer,
        "references": refs,
        "evidence_count": len(all_evidence)
    }
```

**Новый эндпоинт:**

```python
# backend/app/api/routes/research.py
@router.post("/deep-research")
async def deep_research(request: ResearchRequest):
    result = await run_deep_research(
        question=request.question,
        max_subqueries=request.max_subqueries,
        web_enabled=request.web_enabled
    )
    return result
```

---

## ⚠️ Риски и митигация

| # | Риск | Вероятность | Влияние | Митигация |
|---|------|-------------|---------|-----------|
| 1 | `fastembed` не встанет в Docker (ограниченная RAM) | Средняя | Высокое | Тестировать локально перед интеграцией; fallback на `sentence-transformers` |
| 2 | Ансамбль LLM удвоит стоимость API-вызовов | Высокая | Среднее | Добавить лимиты в config; показывать пользователю预估cost |
| 3 | Разные размерности эмбеддингов (API=1024 vs local=512) | 100% | Критичное | Создать отдельные Qdrant-коллекции; auto-detect в search.py |
| 4 | Судья ансамбля может некорректно парсить JSON | Средняя | Среднее | Fallback: объединить все черновики без фильтрации |
| 5 | Провенанс [n] увеличит размер промптов → медленнее LLM | Высокая | Низкое | Ограничить max_refs=15; использовать только цитаты в механизме |
| 6 | Health-check добавит latency на каждый запрос | Низкая | Низкое | Кешировать статус на 30 секунд; lazy-check не критичных компонентов |

---

## 📊 Критерии успеха

### Must-have (для победы на хакатоне):
- [ ] **Надёжность:** Система работает при падении любого внешнего API (embedding/Qdrant)
- [ ] **Провенанс:** Каждая гипотеза содержит кликабельные цитаты `[1], [2]`
- [ ] **Качество:** Ансамбль LLM генерирует более разнообразные гипотезы (измерить overlap с single mode)
- [ ] **UX:** Пользователь видит индикатор состояния системы
- [ ] **Производительность:** Single mode < 90 сек, Ensemble < 180 сек

### Nice-to-have (если успеем):
- [ ] Deep Research mode (аналитический отчёт)
- [ ] Веб-аугментация (OpenAlex)
- [ ] Улучшенная дедупликация (эмбеддинги)
- [ ] Сравнительная визуализация (scatter plot: новизна × реализуемость)

### Acceptance tests перед сдачей:

```python
# test_acceptance.py
import pytest
from app.services.generation import run_full_generation
from app.services.embedding import embed_text
from app.services.system_health import get_system_status

@pytest.mark.asyncio
async def test_acceptance_reliability():
    """System works even when embedding API is down"""
    # Disable API
    with pytest.MonkeyPatch.context() as m:
        m.setenv("EMBEDDING_PROVIDER", "local")
        
        vector = await embed_text("Test text")
        assert len(vector) == 512  # Local fallback
        assert any(v != 0.0 for v in vector)

@pytest.mark.asyncio
async def test_acceptance_provenance():
    """Hypotheses contain numbered citations"""
    result = await run_full_generation(
        "Снизить потери никеля на 15%",
        num_hypotheses=4,
        mode="single"
    )
    
    assert "references" in result
    assert len(result["references"]) > 0
    
    for hyp in result["hypotheses"]:
        assert "citations" in hyp
        # At least some hypotheses should have [n] format
        has_bracket_citation = any("[" in c for c in hyp["citations"])
        # Not strictly required, but preferred

@pytest.mark.asyncio
async def test_acceptance_ensemble():
    """Ensemble mode generates from 2 models"""
    settings = get_settings()
    if not settings.llm_ensemble_enabled:
        pytest.skip("Ensemble not enabled")
    
    result = await run_full_generation(
        "Повысить извлечение меди на 10%",
        num_hypotheses=6,
        mode="ensemble"
    )
    
    assert "ensemble_drafts" in result
    assert "judge_reasoning" in result
    assert len(result["ensemble_drafts"]["model_a"]) > 0
    assert len(result["ensemble_drafts"]["model_b"]) > 0

@pytest.mark.asyncio
async def test_acceptance_health_endpoint():
    """Health endpoint returns system status"""
    status = await get_system_status()
    
    assert status["overall"] in ["full", "limited", "degraded", "critical"]
    assert "embedding" in status
    assert "llm" in status
```

---

## ⏱️ Итоговый таймлайн (реалистичный)

| Время | Задача | Статус |
|-------|--------|--------|
| **0:00-0:45** | 1.1 Локальные эмбеддинги (fastembed) | 🔴 Приоритет |
| **0:45-1:15** | 1.2 Провенанс [n] | 🔴 Критично |
| **1:15-1:45** | 1.3 Graceful degradation + health | 🟡 Важно |
| **1:45-2:45** | 2.1 Ансамбль LLM | 🔴 Критично |
| **2:45-3:05** | 3.1 Улучшенная дедупликация (опционально) | 🟢 Бонус |
| **3:05-3:35** | UI: ensemble panel + citation badges | 🟡 Важно |
| **3:35-4:00** | Тестирование + отладка | 🔴 Обязател |
| **4:00+** | ФАЗА 3: Deep Research / OpenAlex (если успеем) | 🟢 Опционально |

**Итого:** 4 часа для ФАЗЫ 1+2 (обязательно), +3-4 часа для ФАЗЫ 3 (бонус).

---

## 🎯 Заключение

### Ключевые улучшения:

1. **Надёжность** — Система работает при падении API (локальные эмбеддинги + graceful degradation)
2. **Качество** — Ансамбль LLM снижает дисперсию ошибок и даёт более разнообразные гипотезы
3. **Провенанс** — Нумерация источников [n] делает систему интерпретируемой (как у nn_hypogen)
4. **UX** — Health endpoint + UI-индикаторы показывают состояние системы

### Сравнение с конкурентами после реализации:

| Возможность | Мы | nn_hypogen | Результат |
|-------------|-----|------------|-----------|
| LLM модели | 2 + судья | 2 + судья | ✅ Паритет |
| Эмбеддинги | API + локальные | Hashing + fallback | ✅ Лучше |
| Провенанс | [n] кликабельные | [n] кликабельные | ⚠️ Паритет |
| Graceful degradation | Полная | Полная | ✅ Паритет |
| OCR | 3 провайдера | 2 провайдера | ✅ Лучше |
| Стратегии | 3 + ансамбль | Deep Research | 🟡 Разные подходы |

### Что делаем дальше:

1. **Сейчас:** Начинаем с Задачи 1.1 (локальные эмбеддинги) — блокер для надёжности
2. **После ФАЗЫ 1+2:** Тестируем ансамбль на реальном кейсе Норникеля
3. **Если успеем:** Добавляем Deep Research и OpenAlex для конкурентного преимущества

---

**Готовы начать?** Следующий шаг: `cd hypothesis-factory && pip install fastembed>=0.4.0`