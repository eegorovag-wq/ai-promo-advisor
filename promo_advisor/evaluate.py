"""Оценка качества: сравнение версий логики на заранее размеченных сценариях.

Без этого файла получилась бы демонстрация «мы прикрутили LLM». Здесь видно
другое: как решение меняли и чем доказали, что стало лучше.

Разметку делал человек: для каждого сценария указано, какой товар должен быть
первым и почему. Версия v1 — наивный приоритет (остаток, продажи, публикации),
версия v2 — текущая (добавлены маржа, сезонность и явные отказы).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from .models import Product, PromoContext, Recommendation
from .scoring import rank_products

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

Ranker = Callable[[list[Product], PromoContext, int], list[Recommendation]]


def rank_products_v1(products: list[Product], context: PromoContext, limit: int = 3) -> list[Recommendation]:
    """Первая версия: без маржи и сезонности, отказов нет — только баллы.

    Оставлена в репозитории намеренно: на ней считаются цифры «до», и видно,
    какие именно ошибки исправила вторая версия.
    """
    scored: list[Recommendation] = []
    for product in products:
        points = 0
        if product.made_to_order or product.stock >= 30:
            points += 35
        elif product.stock >= 10:
            points += 28
        elif product.stock > 0:
            points += 20
        else:
            points += 10
        points += 25 if product.units_sold_30d == 0 else 12 if product.units_sold_30d <= 5 else 4
        points += 30 if product.posts_30d == 0 else 15 if product.posts_30d == 1 else 0
        scored.append(
            Recommendation(
                product_id=product.product_id,
                title=product.title,
                score=min(100, points),
                factors=(),
                risks=(),
                evidence={
                    "stock": product.stock,
                    "margin_rub": product.margin_rub,
                    "units_sold_30d": product.units_sold_30d,
                    "posts_30d": product.posts_30d,
                },
            )
        )
    scored.sort(key=lambda rec: (-rec.score, rec.title))
    return scored[:limit]


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    expected: str
    got: str | None
    passed: bool
    note: str


@dataclass(frozen=True)
class EvaluationReport:
    version: str
    total: int
    passed: int
    results: list[ScenarioResult]

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def failures(self) -> list[ScenarioResult]:
        return [item for item in self.results if not item.passed]


def load_scenarios(path: str | Path | None = None) -> list[dict]:
    source = Path(path) if path else DATA_DIR / "scenarios.json"
    return json.loads(source.read_text(encoding="utf-8"))["scenarios"]


def _product_from_scenario(item: dict) -> Product:
    return Product(
        product_id=item["product_id"],
        title=item["title"],
        category=item.get("category", "тест"),
        price_rub=float(item["price_rub"]),
        cost_rub=float(item["cost_rub"]),
        stock=int(item["stock"]),
        units_sold_30d=int(item["units_sold_30d"]),
        posts_30d=int(item["posts_30d"]),
        occasions=tuple(item.get("occasions", [])),
        made_to_order=bool(item.get("made_to_order", False)),
    )


def run_evaluation(ranker: Ranker, version: str, scenarios: list[dict] | None = None) -> EvaluationReport:
    """Прогоняет сценарии и считает долю совпадений с экспертной разметкой."""
    scenarios = scenarios or load_scenarios()
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        products = [_product_from_scenario(item) for item in scenario["products"]]
        context = PromoContext(
            today=date.fromisoformat(scenario["today"]),
            occasion_dates={k: date.fromisoformat(v) for k, v in scenario.get("occasions", {}).items()},
        )
        ranked = ranker(products, context, 3)
        got = ranked[0].product_id if ranked else None
        expected = scenario["expected_top"]
        forbidden = set(scenario.get("must_not_recommend", []))
        recommended_ids = {item.product_id for item in ranked}
        violated = sorted(forbidden & recommended_ids)
        passed = got == expected and not violated
        note = scenario["why"]
        if violated:
            note = f"{note} | В выдаче запрещённые товары: {', '.join(violated)}"
        results.append(ScenarioResult(scenario["id"], expected, got, passed, note))
    return EvaluationReport(version, len(results), sum(1 for r in results if r.passed), results)


def compare_versions() -> dict[str, EvaluationReport]:
    scenarios = load_scenarios()
    return {
        "v1": run_evaluation(rank_products_v1, "v1 (без маржи и сезонности)", scenarios),
        "v2": run_evaluation(rank_products, "v2 (текущая)", scenarios),
    }
