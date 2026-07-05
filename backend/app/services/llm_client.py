import httpx
import logging
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import get_settings

logger = logging.getLogger(__name__)


class YandexLLMClient:
    def __init__(self, api_key: str, folder_id: str, api_base: str, model: str):
        self.api_key = api_key
        self.folder_id = folder_id
        self.api_base = api_base
        self.model = model
        self.model_uri = f"gpt://{folder_id}/{model}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        return await self._generate(system_prompt, user_prompt, temperature)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        return await self._generate(system_prompt, user_prompt, temperature)

    async def _generate(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        messages = [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": user_prompt},
        ]
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.api_base}/completion",
                json={
                    "modelUri": self.model_uri,
                    "completionOptions": {
                        "stream": False,
                        "temperature": temperature,
                        "maxTokens": "4000",
                    },
                    "messages": messages,
                },
                headers={"Authorization": f"Api-Key {self.api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            alternatives = data.get("result", {}).get("alternatives", [])
            if alternatives:
                return alternatives[0].get("message", {}).get("text", "")
            raise ValueError(f"Yandex LLM: no alternatives in response: {data}")


class OpenAILLMClient:
    def __init__(self, api_key: str, api_base: str, model: str):
        self.api_base = api_base
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        return await self._generate(system_prompt, user_prompt, temperature)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        return await self._generate(system_prompt, user_prompt, temperature)

    async def _generate(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=4000,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        message_dict = choice.message.model_dump() if hasattr(choice.message, 'model_dump') else {}
        reasoning = message_dict.get('reasoning_content', None) or ""
        logger.info(f"LLM response: content={len(content)} chars, reasoning={len(reasoning)} chars, model={self.model}, keys={list(message_dict.keys())}")
        if len(content) < 100:
            logger.warning(f"LLM short response. Content: '{content[:150]}'. Message keys: {list(message_dict.keys())}")
        return content


def get_llm_client(model: str = None):
    s = get_settings()
    if s.llm_provider == "openai":
        return OpenAILLMClient(
            api_key=s.llm_api_key,
            api_base=s.llm_api_base,
            model=model or s.llm_model,
        )
    return YandexLLMClient(
        api_key=s.yandex_api_key,
        folder_id=s.yandex_folder_id,
        api_base=s.yandex_api_base,
        model=model or s.yandex_model,
    )
