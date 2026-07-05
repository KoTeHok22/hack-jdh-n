import asyncio
import json
import logging
import re
from pathlib import Path
from app.services.search import hybrid_search, build_context
from app.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_prompts_path = Path(__file__).parent / "prompts.json"
with open(_prompts_path, "r", encoding="utf-8") as f:
    _prompts = json.load(f)

SYSTEM_PROMPT_ANALYSIS = _prompts["system_prompt_analysis"]
SYSTEM_PROMPT_GENERATION = _prompts["system_prompt_generation"]
GENERATION_STRATEGY_ANALOGY = _prompts["generation_strategy_analogy"]
GENERATION_STRATEGY_GAP = _prompts["generation_strategy_gap"]
GENERATION_STRATEGY_CROSS_DOMAIN = _prompts["generation_strategy_cross_domain"]
SYSTEM_PROMPT_SCORING = _prompts["system_prompt_scoring"]
HYPOTHESIS_OUTPUT_FORMAT = _prompts["hypothesis_output_format"]
SCORING_OUTPUT_FORMAT = _prompts["scoring_output_format"]


def _extract_json(text: str) -> list | dict:
    logger.debug(f"Attempting to parse JSON from {len(text)} chars")
    
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if json_match:
        try:
            result = json.loads(json_match.group(1).strip())
            logger.debug(f"Parsed JSON from markdown block: {type(result)}")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse markdown JSON: {e}")

    bracket_match = re.search(r'[\[{][\s\S]*[\]}]', text)
    if bracket_match:
        raw = bracket_match.group(0)
        try:
            result = json.loads(raw)
            logger.debug(f"Parsed JSON from brackets: {type(result)}")
            return result
        except json.JSONDecodeError:
            cleaned = re.sub(r',(\s*[\]}])', r'\1', raw)
            try:
                result = json.loads(cleaned)
                logger.debug(f"Parsed cleaned JSON from brackets: {type(result)}")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse bracket JSON: {e}")

    try:
        result = json.loads(text)
        logger.debug(f"Parsed raw JSON: {type(result)}")
        return result
    except json.JSONDecodeError:
        cleaned = re.sub(r',(\s*[\]}])', r'\1', text)
        result = json.loads(cleaned)
        logger.debug(f"Parsed cleaned raw JSON: {type(result)}")
        return result


def calculate_composite_score(novelty: float, feasibility: float, impact: float, risk: float) -> float:
    return round(0.25 * novelty + 0.25 * feasibility + 0.30 * impact - 0.20 * risk, 2)


async def parse_problem(statement: str, context: str, model: str = None) -> dict:
    llm = get_llm_client(model=model)
    prompt = f"""Проанализируй проблему и извлеки структурированные данные.

ПРОБЛЕМА: {statement}

КОНТЕКСТ:
{context}

Верни JSON:
{{
  "target_kpi": "целевой показатель",
  "target_delta": "желаемое изменение",
  "domain": "домен (metallurgy/flotation/leaching/etc)",
  "key_constraints": ["ограничение 1", "ограничение 2"],
  "priority_areas": ["область 1", "область 2"]
}}"""

    result = await llm.generate_json(SYSTEM_PROMPT_ANALYSIS, prompt)
    return _extract_json(result)


def _normalize(text: str) -> set[str]:
    return set(re.findall(r'\w+', text.lower()))


def _hypothesis_similarity(a: str, b: str) -> float:
    set_a = _normalize(a)
    set_b = _normalize(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def _deduplicate_hypotheses(hypotheses: list[dict], threshold: float = 0.85) -> list[dict]:
    unique = []
    for hyp in hypotheses:
        statement = hyp.get("statement", "")
        is_dup = False
        for existing in unique:
            sim = _hypothesis_similarity(statement, existing.get("statement", ""))
            if sim > threshold:
                is_dup = True
                logger.debug(f"Duplicate removed (sim={sim:.2f}): {statement[:60]}...")
                break
        if not is_dup:
            unique.append(hyp)
    return unique


async def _gen_strategy(
    llm,
    statement: str,
    parsed_problem: dict,
    context: str,
    strategy_prompt: str,
    count: int,
    existing_text: str,
    strategy_name: str = "",
) -> list[dict]:
    final_prompt = strategy_prompt
    if "{existing}" in strategy_prompt:
        final_prompt = strategy_prompt.format(existing=existing_text)

    user_prompt = f"""ПРОБЛЕМА: {statement}

АНАЛИЗ ПРОБЛЕМЫ:
{json.dumps(parsed_problem, ensure_ascii=False, indent=2)}

КОНТЕКСТ ИЗ ИСТОЧНИКОВ:
{context}

{final_prompt}

ВАЖНО: Сгенерируй ровно {count} УНИКАЛЬНЫХ гипотез. Каждая гипотеза должна быть принципиально разной.

{HYPOTHESIS_OUTPUT_FORMAT}"""

    result = await llm.generate_json(SYSTEM_PROMPT_GENERATION, user_prompt, temperature=0.7)
    try:
        hypotheses = _extract_json(result)
        if isinstance(hypotheses, list):
            logger.info(f"[{strategy_name}] Generated {len(hypotheses)}/{count} hypotheses")
            return hypotheses
        logger.warning(f"[{strategy_name}] Expected list, got {type(hypotheses)}")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"[{strategy_name}] Failed to parse JSON: {e}")
    return []


async def generate_hypotheses(
    statement: str,
    parsed_problem: dict,
    context: str,
    num_hypotheses: int = 8,
    model: str = None,
) -> list[dict]:
    llm = get_llm_client(model=model)

    oversample = 1.5
    total_requested = int(num_hypotheses * oversample)
    
    analogy_count = int(total_requested * 0.4)
    gap_count = int(total_requested * 0.4)
    cross_count = total_requested - analogy_count - gap_count

    analogy_task = _gen_strategy(llm, statement, parsed_problem, context, GENERATION_STRATEGY_ANALOGY, analogy_count, "", "analogy")
    gap_task = _gen_strategy(llm, statement, parsed_problem, context, GENERATION_STRATEGY_GAP, gap_count, "", "gap")

    analogy_hyps, gap_hyps = await asyncio.gather(analogy_task, gap_task)

    existing_lines = []
    for h in analogy_hyps:
        existing_lines.append(f"- {h.get('statement', '')}")
    for h in gap_hyps:
        existing_lines.append(f"- {h.get('statement', '')}")
    existing_text = "\n".join(existing_lines)

    cross_hyps = await _gen_strategy(llm, statement, parsed_problem, context, GENERATION_STRATEGY_CROSS_DOMAIN, cross_count, existing_text, "cross")

    all_hypotheses = analogy_hyps + gap_hyps + cross_hyps
    logger.info(f"Total before dedup: {len(all_hypotheses)} (analogy={len(analogy_hyps)}, gap={len(gap_hyps)}, cross={len(cross_hyps)})")
    
    deduplicated = _deduplicate_hypotheses(all_hypotheses)
    logger.info(f"After dedup: {len(deduplicated)}")
    
    if len(deduplicated) < num_hypotheses:
        missing = num_hypotheses - len(deduplicated)
        logger.warning(f"Only {len(deduplicated)} hypotheses after dedup, requesting {missing} more...")
        fallback_hyps = await _gen_strategy(
            llm, statement, parsed_problem, context,
            GENERATION_STRATEGY_GAP, missing, existing_text, "fallback"
        )
        deduplicated.extend(fallback_hyps)
        deduplicated = _deduplicate_hypotheses(deduplicated)
        logger.info(f"After fallback: {len(deduplicated)}")
    
    return deduplicated[:num_hypotheses]


async def score_hypotheses(hypotheses: list[dict], statement: str, context: str, model: str = None) -> list[dict]:
    llm = get_llm_client(model=model)

    hypotheses_text = json.dumps(
        [{"index": i, "statement": h.get("statement", ""), "mechanism": h.get("mechanism", "")}
         for i, h in enumerate(hypotheses)],
        ensure_ascii=False, indent=2
    )

    user_prompt = f"""ПРОБЛЕМА: {statement}

ГИПОТЕЗЫ:
{hypotheses_text}

{SCORING_OUTPUT_FORMAT}"""

    result = await llm.generate_json(SYSTEM_PROMPT_SCORING, user_prompt, temperature=0.1)
    try:
        scores = _extract_json(result)
        if isinstance(scores, list):
            return scores
    except (json.JSONDecodeError, ValueError):
        pass

    return [
        {
            "index": i,
            "novelty": 3,
            "feasibility": 3,
            "impact": 3,
            "risk": 3,
            "confidence": 0.5,
            "risks": [],
            "verification_plan": "",
        }
        for i in range(len(hypotheses))
    ]


async def run_full_generation(
    statement: str,
    document_ids: list = None,
    num_hypotheses: int = 8,
    model: str = None,
) -> dict:
    search_results = await hybrid_search(statement, top_k=15)
    context = build_context(search_results)

    parsed_problem = await parse_problem(statement, context, model=model)

    hypotheses = await generate_hypotheses(statement, parsed_problem, context, num_hypotheses, model=model)

    if not hypotheses:
        return {"parsed_problem": parsed_problem, "hypotheses": [], "context_chunks": search_results}

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
        })

    scored_hypotheses.sort(key=lambda x: x["composite_score"], reverse=True)

    return {
        "parsed_problem": parsed_problem,
        "hypotheses": scored_hypotheses,
        "context_chunks": search_results,
    }
