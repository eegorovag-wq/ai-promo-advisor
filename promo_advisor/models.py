"""Типы данных приложения.

Намеренно обычные dataclass-ы из стандартной библиотеки: никаких зависимостей,
всё видно и проверяется без магии.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any


@dataclass(frozen=True)
class Product:
    """Товар каталога в том виде, в каком его отдаёт магазин."""

    product_id: str
    title: str
    category: str
    price_rub: float
    cost_rub: float            # себестоимость: закупка + упаковка
    stock: int                 # свободный остаток (уже за вычетом резервов)
    units_sold_30d: int        # оплаченные продажи за 30 дней, без возвратов
    posts_30d: int             # сколько раз товар уже публиковали за 30 дней
    occasions: tuple[str, ...] = ()   # к каким поводам относится товар
    made_to_order: bool = False       # изготавливается под заказ: остаток не ограничивает

    @property
    def margin_rub(self) -> float:
        return round(self.price_rub - self.cost_rub, 2)

    @property
    def margin_share(self) -> float:
        """Доля маржи в цене. 0.0, если цена нулевая — чтобы не делить на ноль."""
        if self.price_rub <= 0:
            return 0.0
        return self.margin_rub / self.price_rub


@dataclass(frozen=True)
class Factor:
    """Один вклад в итоговый приоритет: почему товар поднялся или опустился."""

    code: str
    label: str
    points: int


@dataclass(frozen=True)
class Recommendation:
    """Результат расчёта по одному товару.

    score — не прогноз спроса и не ожидаемая прибыль, а приоритет проверки
    человеком. Так и подписано во всех ответах, чтобы никто не принял его за EV.
    """

    product_id: str
    title: str
    score: int
    factors: tuple[Factor, ...]
    risks: tuple[str, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["factors"] = [asdict(f) for f in self.factors]
        data["risks"] = list(self.risks)
        return data


@dataclass(frozen=True)
class PromoContext:
    """Внешние условия на момент расчёта."""

    today: date
    occasion_dates: dict[str, date] = field(default_factory=dict)  # повод -> дата праздника
