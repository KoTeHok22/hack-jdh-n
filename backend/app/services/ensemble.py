import asyncio
import logging
from app.services.llm_client import get_llm_client, YandexLLMClient, OpenAILLMClient
from app.services.generation import _gen_strategy
from app.config import get_settings

logger = logging.getLogger(__name__)


def _get_secondary_llm_client():
    settings = get_settings()
    if settings.llm_provider_2 == "yandex":
        return YandexLLMClient(
            api_key=settings.yandex_api_key,
            folder_id=settings.yandex_folder_id,
            api_base=settings.yandex_api_base,
            model=settings.yandex_model,
        )
    return OpenAILLMClient(
        api_key=settings.llm_api_key_2,
        api_base=settings.llm_api_base_2,
        model=settings.llm_model_2,
    )


async def ensemble_generate(
    statement: str,
    parsed_problem: dict,
    context: str,
    num_hypotheses: int = 8,
    model: str = None,
) -> list[dict]:
    settings = get_settings()
    if not settings.llm_ensemble_enabled:
        logger.warning("Ensemble is disabled, falling back to single LLM")
        from app.services.generation import generate_hypotheses
        return await generate_hypotheses(statement, parsed_problem, context, num_hypotheses, model)
    
    primary_llm = get_llm_client(model)
    secondary_llm = _get_secondary_llm_client()
    
    from app.services.generation import (
        GENERATION_STRATEGY_ANALOGY,
        GENERATION_STRATEGY_GAP,
        GENERATION_STRATEGY_CROSS_DOMAIN,
    )
    
    analogy_count = int(num_hypotheses * 0.4)
    gap_count = int(num_hypotheses * 0.4)
    cross_count = num_hypotheses - analogy_count - gap_count
    
    primary_analogy = _gen_strategy(primary_llm, statement, parsed_problem, context,
                                     GENERATION_STRATEGY_ANALOGY, analogy_count, "", "primary-analogy")
    primary_gap = _gen_strategy(primary_llm, statement, parsed_problem, context,
                                 GENERATION_STRATEGY_GAP, gap_count, "", "primary-gap")
    
    secondary_analogy = _gen_strategy(secondary_llm, statement, parsed_problem, context,
                                       GENERATION_STRATEGY_ANALOGY, analogy_count, "", "secondary-analogy")
    secondary_gap = _gen_strategy(secondary_llm, statement, parsed_problem, context,
                                   GENERATION_STRATEGY_GAP, gap_count, "", "secondary-gap")
    
    p_analogy, p_gap, s_analogy, s_gap = await asyncio.gather(
        primary_analogy, primary_gap, secondary_analogy, secondary_gap
    )
    
    existing_text = "\n".join([
        f"- {h.get('statement', '')}" 
        for h in p_analogy + p_gap + s_analogy + s_gap
    ])
    
    primary_cross = _gen_strategy(primary_llm, statement, parsed_problem, context,
                                   GENERATION_STRATEGY_CROSS_DOMAIN, cross_count // 2,
                                   existing_text, "primary-cross")
    secondary_cross = _gen_strategy(secondary_llm, statement, parsed_problem, context,
                                     GENERATION_STRATEGY_CROSS_DOMAIN, cross_count - cross_count // 2,
                                     existing_text, "secondary-cross")
    
    p_cross, s_cross = await asyncio.gather(primary_cross, secondary_cross)
    
    all_hypotheses = p_analogy + p_gap + p_cross + s_analogy + s_gap + s_cross
    
    logger.info(f"Ensemble generated {len(all_hypotheses)} total hypotheses "
                f"(primary: {len(p_analogy)+len(p_gap)+len(p_cross)}, "
                f"secondary: {len(s_analogy)+len(s_gap)+len(s_cross)})")
    
    from app.services.generation import _deduplicate_hypotheses
    unique = _deduplicate_hypotheses(all_hypotheses, threshold=0.85)
    
    if len(unique) > num_hypotheses:
        unique = unique[:num_hypotheses]
    
    logger.info(f"Ensemble result: {len(unique)} unique hypotheses after dedup")
    
    return unique
