"""Слой языковой модели.

Два важных решения:

1. Модель отвечает строгой структурой (JSON по схеме), а не свободным текстом.
   Ответ проверяется кодом; если он не проходит проверку, он отбрасывается —
   «почти правильный» JSON в бизнес-процесс не попадает.
2. По умолчанию работает заглушка без сети и без ключа. Репозиторий можно
   склонировать и запустить где угодно, а тесты не зависят от внешнего сервиса.
   Реальный провайдер подключается переменными окружения.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .models import Recommendation

MAX_TEXT_LENGTH = 420


@dataclass(frozen=True)
class DraftRequest:
    recommendation: Recommendation
    channel: str               # telegram | max
    audience: str              # для кого пишем: «постоянные клиенты», «новые» и т.д.


@dataclass(frozen=True)
class Draft:
    """Результат работы модели после проверки структуры."""

    headline: str
    body: str
    reason: str                # объяснение выбора товара для менеджера
    call_to_action: str

    @property
    def full_text(self) -> str:
        return f"{self.headline}\n\n{self.body}\n\n{self.call_to_action}"


class LlmError(RuntimeError):
    pass


class LlmProvider(Protocol):
    def generate_draft(self, request: DraftRequest) -> Draft: ...


SYSTEM_PROMPT = (
    "Ты пишешь короткие публикации для магазина подарков в спокойном, сдержанном тоне. "
    "Запрещено: обещать выгоду или прибыль, давить срочностью, выдумывать скидки, "
    "характеристики и наличие. Пиши только то, что есть во входных данных. "
    "Ответ строго в JSON: {\"headline\": str, \"body\": str, \"reason\": str, \"call_to_action\": str}. "
    f"headline до 60 символов, body до {MAX_TEXT_LENGTH} символов, без эмодзи."
)


def build_user_prompt(request: DraftRequest) -> str:
    """Промпт собирается из посчитанных фактов, а не из свободного пересказа.

    Так модель объясняет уже принятое решение и не может подставить свои цифры.
    """
    rec = request.recommendation
    facts = "\n".join(f"- {factor.label}" for factor in rec.factors)
    return (
        f"Товар: {rec.title}\n"
        f"Канал: {request.channel}\n"
        f"Аудитория: {request.audience}\n"
        f"Факты для объяснения:\n{facts}\n"
        "Задача: заголовок, короткий текст публикации, объяснение выбора для менеджера "
        "и призыв к действию без давления."
    )


def parse_draft(raw: str) -> Draft:
    """Разбор и проверка ответа модели. Любое отклонение — отказ, а не починка."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise LlmError(f"Ответ модели не является JSON: {error}") from error
    if not isinstance(data, dict):
        raise LlmError("Ответ модели не является объектом JSON")
    required = ("headline", "body", "reason", "call_to_action")
    unexpected = sorted(set(data) - set(required))
    if unexpected:
        raise LlmError(f"В ответе модели есть лишние поля: {', '.join(unexpected)}")
    missing = [key for key in required if not isinstance(data.get(key), str) or not data[key].strip()]
    if missing:
        raise LlmError(f"В ответе модели нет обязательных полей: {', '.join(missing)}")
    if len(data["headline"]) > 60:
        raise LlmError("Заголовок длиннее 60 символов")
    if len(data["body"]) > MAX_TEXT_LENGTH:
        raise LlmError(f"Текст длиннее {MAX_TEXT_LENGTH} символов")
    return Draft(
        headline=data["headline"].strip(),
        body=data["body"].strip(),
        reason=data["reason"].strip(),
        call_to_action=data["call_to_action"].strip(),
    )


class StubLlm:
    """Заглушка без сети: собирает текст из тех же фактов по шаблону.

    Нужна не для красоты: на ней прогоняются тесты и оценка качества, поэтому
    результаты воспроизводимы и не стоят денег.
    """

    def generate_draft(self, request: DraftRequest) -> Draft:
        rec = request.recommendation
        reason = "; ".join(factor.label for factor in rec.factors[:3])
        payload = {
            "headline": rec.title[:60],
            "body": (
                f"{rec.title} — из подборки для аудитории «{request.audience}». "
                f"Собираем подарок под ваш повод, упакуем и подпишем открытку."
            )[:MAX_TEXT_LENGTH],
            "reason": f"Товар выбран по данным магазина: {reason}.",
            "call_to_action": "Напишите нам, подберём и соберём подарок.",
        }
        return parse_draft(json.dumps(payload, ensure_ascii=False))


class LegacyPromptStub:
    """Как выглядел текст до правки промпта: рекламные штампы, давление срочностью,
    выдуманная скидка и длинное полотно. Нужна, чтобы сравнение версий промпта
    считалось числом, а не вспоминалось на словах."""

    def generate_draft(self, request: DraftRequest) -> Draft:
        rec = request.recommendation
        payload = {
            "headline": f"Только сегодня! {rec.title} по лучшей цене — успей купить",
            "body": (
                f"Спешите: {rec.title} со скидкой 50 процентов, дешевле не найдёшь нигде! "
                "Мы гарантируем прибыль вашему празднику и дарим бесплатно упаковку. "
                "Последний шанс забрать подарок мечты по такой цене, осталось всего 3 штуки, "
                "успейте оформить заказ прямо сейчас, пока предложение действует."
            ),
            "reason": "Товар выбран, потому что он отличный.",
            "call_to_action": "Срочно пишите нам!",
        }
        return Draft(
            headline=payload["headline"],
            body=payload["body"],
            reason=payload["reason"],
            call_to_action=payload["call_to_action"],
        )


class OpenAiCompatibleLlm:
    """Адаптер к любому OpenAI-совместимому API (OpenAI, Яндекс через шлюз, локальная модель).

    Ключ и адрес берутся из окружения и никогда не попадают в репозиторий.
    """

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenAiCompatibleLlm | None":
        api_key = os.getenv("PROMO_LLM_API_KEY")
        if not api_key:
            return None
        return cls(
            base_url=os.getenv("PROMO_LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=api_key,
            model=os.getenv("PROMO_LLM_MODEL", "gpt-4o-mini"),
        )

    def generate_draft(self, request: DraftRequest) -> Draft:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0.4,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(request)},
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as error:  # сеть, таймаут, 4xx/5xx
            raise LlmError(f"Запрос к модели не удался: {error}") from error
        try:
            raw = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LlmError("Неожиданная структура ответа модели") from error
        return parse_draft(raw)


def get_provider() -> LlmProvider:
    """Реальная модель, если задан ключ, иначе заглушка. Без ключа всё тоже работает."""
    return OpenAiCompatibleLlm.from_env() or StubLlm()
