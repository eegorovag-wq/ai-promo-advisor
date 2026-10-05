"""Ограничения, которые нельзя обойти языковой модели.

Правило проекта: модель пишет текст и объясняет выбор, но ничего не решает сама.
Всё, что меняет деньги, остатки или публикуется наружу, проходит через эти проверки
и через согласование человеком.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

# Максимальная скидка, которую вообще можно упомянуть в публикации.
MAX_DISCOUNT_PERCENT = 10
# Тихие часы: ночью не публикуем, даже если всё согласовано.
QUIET_FROM = time(22, 0)
QUIET_TO = time(9, 0)
# Слова, которых не должно быть в брендовом тексте.
BANNED_PHRASES = (
    "скидка 50",
    "только сегодня",
    "успей купить",
    "дешевле не найдёшь",
    "бесплатно",
    "гарантируем прибыль",
)


class GuardrailError(RuntimeError):
    """Нарушение правила. Текст сообщения показывается менеджеру как есть."""


@dataclass(frozen=True)
class PublishPolicy:
    """Настройки, которые владелец задаёт в интерфейсе, а не модель в промпте."""

    kill_switch: bool = False          # общий стоп: выключает публикации целиком
    max_discount_percent: int = MAX_DISCOUNT_PERCENT


def check_text(text: str) -> None:
    """Проверяет готовый текст публикации на запрещённые обещания."""
    lowered = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            raise GuardrailError(f"В тексте запрещённая формулировка: «{phrase}»")


def check_discount(discount_percent: int, policy: PublishPolicy) -> None:
    if discount_percent < 0:
        raise GuardrailError("Скидка не может быть отрицательной")
    if discount_percent > policy.max_discount_percent:
        raise GuardrailError(
            f"Скидка {discount_percent}% выше разрешённой {policy.max_discount_percent}%"
        )


def check_quiet_hours(moment: datetime) -> None:
    """Тихие часы считаются корректно и для интервала через полночь."""
    now = moment.time()
    in_quiet = now >= QUIET_FROM or now < QUIET_TO
    if in_quiet:
        raise GuardrailError(
            f"Тихие часы с {QUIET_FROM:%H:%M} до {QUIET_TO:%H:%M}: публикация отложена"
        )


def check_can_publish(
    *,
    approved_by: str | None,
    text: str,
    discount_percent: int,
    moment: datetime,
    policy: PublishPolicy,
) -> None:
    """Единственная дверь к публикации. Падает при первом же нарушении.

    Порядок проверок — от самого грубого запрета к частному, чтобы сообщение
    менеджеру было про главную причину, а не про мелочь.
    """
    if policy.kill_switch:
        raise GuardrailError("Включён общий стоп публикаций")
    if not approved_by:
        raise GuardrailError("Публикация без согласования человеком запрещена")
    check_discount(discount_percent, policy)
    check_text(text)
    check_quiet_hours(moment)
