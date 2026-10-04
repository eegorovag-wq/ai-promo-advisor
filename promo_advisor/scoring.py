"""Детерминированный расчёт приоритета продвижения.

Главное решение всего проекта: приоритет считает обычный код, а не языковая модель.
Модель умеет убедительно объяснять любое число, но проверить его нельзя, а здесь
каждый балл виден, воспроизводим и покрыт тестами. Языковой модели остаётся то,
что она делает хорошо: человеческая формулировка и текст публикации.

Шкала намеренно огрублена до понятных ступеней: менеджеру важно «почему этот
товар», а не разница в третьем знаке. Но ступени разнесены так, чтобы товары не
слипались в одинаковые 90 из 100 — иначе порядок начинает решать алфавит.
"""

from __future__ import annotations

from .models import Factor, Product, PromoContext, Recommendation

MAX_SCORE = 100
# Ближе этого числа дней до праздника повод считается «ближайшим».
OCCASION_HORIZON_DAYS = 45


def _stock_factor(product: Product) -> tuple[Factor, list[str]]:
    """Нечего отгружать — нечего и продвигать. Товар под заказ остатком не ограничен."""
    if product.made_to_order:
        return Factor("stock_made_to_order", "Под заказ: остаток не ограничивает", 18), []
    if product.stock >= 30:
        return Factor("stock_high", f"Остаток {product.stock} шт.", 25), []
    if product.stock >= 10:
        return Factor("stock_ok", f"Остаток {product.stock} шт.", 18), []
    return (
        Factor("stock_low", f"Остаток только {product.stock} шт.", 6),
        ["Низкий остаток: спрос после публикации может быть нечем закрыть"],
    )


def _margin_factor(product: Product) -> tuple[Factor, list[str]]:
    """Продвигать выгоднее то, что приносит деньги, а не то, что просто лежит."""
    share = product.margin_share
    if share >= 0.45:
        return Factor("margin_high", f"Маржа {share:.0%} ({product.margin_rub:.0f} ₽)", 25), []
    if share >= 0.30:
        return Factor("margin_ok", f"Маржа {share:.0%} ({product.margin_rub:.0f} ₽)", 16), []
    if share > 0:
        return (
            Factor("margin_low", f"Низкая маржа {share:.0%} ({product.margin_rub:.0f} ₽)", 5),
            ["Низкая маржа: реклама может съесть всю прибыль"],
        )
    return (
        Factor("margin_negative", f"Маржа отрицательная ({product.margin_rub:.0f} ₽)", 0),
        ["Цена ниже себестоимости: продвигать нельзя, сначала разберитесь с ценой"],
    )


def _sales_factor(product: Product) -> tuple[Factor, list[str]]:
    """Ноль продаж — повод присмотреться, но это сигнал, а не доказательство спроса."""
    if product.units_sold_30d == 0:
        return (
            Factor("sales_none", "За 30 дней продаж не было", 18),
            ["Продаж не было: причина может быть в товаре или цене, а не в отсутствии рекламы"],
        )
    if product.units_sold_30d <= 3:
        return Factor("sales_low", f"За 30 дней продано {product.units_sold_30d} шт.", 20), []
    if product.units_sold_30d <= 10:
        return Factor("sales_moderate", f"За 30 дней продано {product.units_sold_30d} шт.", 12), []
    return Factor("sales_active", f"Хорошо продаётся: {product.units_sold_30d} шт. за 30 дней", 5), []


def _promotion_factor(product: Product) -> tuple[Factor, list[str]]:
    """Один и тот же товар в ленте подряд выжигает аудиторию."""
    if product.posts_30d == 0:
        return Factor("not_promoted", "За 30 дней не публиковался", 20), []
    if product.posts_30d == 1:
        return Factor("promoted_once", "За 30 дней была одна публикация", 10), []
    return (
        Factor("promoted_often", f"За 30 дней публикаций: {product.posts_30d}", 0),
        [f"Товар уже публиковали {product.posts_30d} раза за 30 дней"],
    )


def _occasion_factor(product: Product, context: PromoContext) -> tuple[Factor | None, list[str]]:
    """Сезонность: подарок к 8 Марта в феврале стоит дороже, чем он же в июле."""
    best_days: int | None = None
    best_name = ""
    for occasion in product.occasions:
        occasion_date = context.occasion_dates.get(occasion)
        if occasion_date is None:
            continue
        days = (occasion_date - context.today).days
        if days < 0:
            continue
        if best_days is None or days < best_days:
            best_days, best_name = days, occasion
    if best_days is None or best_days > OCCASION_HORIZON_DAYS:
        return None, []
    if best_days <= 14:
        return Factor("occasion_now", f"До повода «{best_name}» {best_days} дн.", 12), []
    return Factor("occasion_soon", f"Повод «{best_name}» через {best_days} дн.", 7), []


def score_product(product: Product, context: PromoContext) -> Recommendation | None:
    """Считает приоритет одного товара.

    Возвращает None, если товар продвигать нельзя в принципе: нет остатка
    у обычного товара или цена ниже себестоимости. Это не «ноль баллов», а отказ:
    такие товары не должны попадать в список даже последними.
    """
    if not product.made_to_order and product.stock <= 0:
        return None
    if product.margin_rub < 0:
        return None

    factors: list[Factor] = []
    risks: list[str] = []
    for factor, factor_risks in (
        _stock_factor(product),
        _margin_factor(product),
        _sales_factor(product),
        _promotion_factor(product),
    ):
        factors.append(factor)
        risks.extend(factor_risks)

    occasion_factor, occasion_risks = _occasion_factor(product, context)
    if occasion_factor is not None:
        factors.append(occasion_factor)
    risks.extend(occasion_risks)

    score = min(MAX_SCORE, sum(factor.points for factor in factors))
    return Recommendation(
        product_id=product.product_id,
        title=product.title,
        score=score,
        factors=tuple(factors),
        risks=tuple(risks),
        evidence={
            "stock": None if product.made_to_order else product.stock,
            "margin_rub": product.margin_rub,
            "margin_share": round(product.margin_share, 4),
            "units_sold_30d": product.units_sold_30d,
            "posts_30d": product.posts_30d,
            "occasions": list(product.occasions),
        },
    )


def rank_products(products: list[Product], context: PromoContext, limit: int = 3) -> list[Recommendation]:
    """Считает и сортирует. При равных баллах порядок задаётся явно, а не алфавитом:
    сначала более прибыльный товар, затем тот, что давно не публиковался."""
    scored = [item for item in (score_product(p, context) for p in products) if item is not None]
    scored.sort(
        key=lambda rec: (
            -rec.score,
            -float(rec.evidence["margin_rub"]),
            rec.evidence["posts_30d"],
            rec.title,
        )
    )
    return scored[: max(0, limit)]
