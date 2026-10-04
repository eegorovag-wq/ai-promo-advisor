"""Загрузка каталога и календаря поводов из JSON.

Данные синтетические и лежат в репозитории: проект должен запускаться у любого,
кто его склонировал, без доступа к чужой базе.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .models import Product, PromoContext

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_products(path: str | Path | None = None) -> list[Product]:
    source = Path(path) if path else DATA_DIR / "catalog.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    return [
        Product(
            product_id=item["product_id"],
            title=item["title"],
            category=item["category"],
            price_rub=float(item["price_rub"]),
            cost_rub=float(item["cost_rub"]),
            stock=int(item["stock"]),
            units_sold_30d=int(item["units_sold_30d"]),
            posts_30d=int(item["posts_30d"]),
            occasions=tuple(item.get("occasions", [])),
            made_to_order=bool(item.get("made_to_order", False)),
        )
        for item in raw["products"]
    ]


def load_context(today: date | None = None, path: str | Path | None = None) -> PromoContext:
    source = Path(path) if path else DATA_DIR / "occasions.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    occasion_dates = {name: date.fromisoformat(value) for name, value in raw["occasions"].items()}
    return PromoContext(today=today or date.today(), occasion_dates=occasion_dates)
