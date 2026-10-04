"""Проверка текста публикации по правилам.

Оценивать текст «на глаз» нельзя: мнение меняется от настроения, а сравнить две
версии промпта нужно числом. Здесь собраны проверки, которые дают одинаковый
ответ при каждом запуске и понятны менеджеру без объяснений.

Главная из них — последняя: в тексте не должно быть чисел, которых нет во
входных данных. Так ловится выдуманная цена, выдуманный остаток и выдуманная
скидка, то есть самая опасная ошибка языковой модели в рекламе.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .guardrails import BANNED_PHRASES
from .llm import MAX_TEXT_LENGTH, Draft
from .models import Recommendation

MIN_BODY_LENGTH = 40
MAX_HEADLINE_LENGTH = 60
PRESSURE_WORDS = ("срочно", "последний шанс", "спешите", "только сейчас", "успейте")


@dataclass(frozen=True)
class TextIssue:
    code: str
    message: str


def _numbers(text: str) -> set[str]:
    """Числа без разделителей: 1 490 ₽ и 1490 ₽ — одно и то же число."""
    return {match.replace(" ", "").replace(" ", "") for match in re.findall(r"\d[\d  ]*", text)}


def check_draft(draft: Draft, recommendation: Recommendation) -> list[TextIssue]:
    """Возвращает список нарушений. Пустой список — текст пригоден к показу человеку."""
    issues: list[TextIssue] = []
    text = draft.full_text
    lowered = text.lower()

    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            issues.append(TextIssue("banned_phrase", f"Запрещённая формулировка: «{phrase}»"))
    for word in PRESSURE_WORDS:
        if word in lowered:
            issues.append(TextIssue("pressure", f"Давление срочностью: «{word}»"))
    if len(draft.headline) > MAX_HEADLINE_LENGTH:
        issues.append(TextIssue("headline_too_long", f"Заголовок длиннее {MAX_HEADLINE_LENGTH} символов"))
    if len(draft.body) > MAX_TEXT_LENGTH:
        issues.append(TextIssue("body_too_long", f"Текст длиннее {MAX_TEXT_LENGTH} символов"))
    if len(draft.body) < MIN_BODY_LENGTH:
        issues.append(TextIssue("body_too_short", "Текст слишком короткий, читателю не за что зацепиться"))
    if not draft.call_to_action.strip():
        issues.append(TextIssue("no_call_to_action", "Нет призыва к действию"))
    if recommendation.title.split(",")[0][:20].lower() not in lowered:
        issues.append(TextIssue("product_not_mentioned", "В тексте не назван сам товар"))

    known = _numbers(" ".join(factor.label for factor in recommendation.factors) + " " + recommendation.title)
    for number in _numbers(text):
        if number not in known and len(number) > 1:
            issues.append(TextIssue("invented_number", f"Число {number} отсутствует во входных данных"))
    return issues
