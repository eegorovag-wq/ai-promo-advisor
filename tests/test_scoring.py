"""Тесты приоритета: проверяем решения, а не внутренние детали расчёта."""

import unittest
from datetime import date

from promo_advisor.models import Product, PromoContext
from promo_advisor.scoring import rank_products, score_product

TODAY = date(2026, 10, 5)
CONTEXT = PromoContext(today=TODAY, occasion_dates={"Новый год": date(2026, 12, 31)})


def product(**overrides) -> Product:
    base = dict(
        product_id="p1",
        title="Товар",
        category="тест",
        price_rub=2000.0,
        cost_rub=1000.0,
        stock=20,
        units_sold_30d=1,
        posts_30d=0,
    )
    base.update(overrides)
    return Product(**base)


class ScoringTest(unittest.TestCase):
    def test_товар_без_остатка_не_рекомендуется(self):
        self.assertIsNone(score_product(product(stock=0), CONTEXT))

    def test_цена_ниже_себестоимости_не_рекомендуется(self):
        self.assertIsNone(score_product(product(price_rub=900, cost_rub=1200), CONTEXT))

    def test_услуга_под_заказ_рекомендуется_без_остатка(self):
        result = score_product(product(stock=0, made_to_order=True), CONTEXT)
        self.assertIsNotNone(result)
        self.assertIsNone(result.evidence["stock"])

    def test_низкий_остаток_даёт_предупреждение(self):
        result = score_product(product(stock=3), CONTEXT)
        self.assertTrue(any("остаток" in risk.lower() for risk in result.risks))

    def test_высокая_маржа_поднимает_приоритет(self):
        rich = score_product(product(product_id="rich", price_rub=4000, cost_rub=1800), CONTEXT)
        poor = score_product(product(product_id="poor", price_rub=1000, cost_rub=900), CONTEXT)
        self.assertGreater(rich.score, poor.score)

    def test_близкий_повод_поднимает_приоритет(self):
        context = PromoContext(today=date(2026, 12, 20), occasion_dates={"Новый год": date(2026, 12, 31)})
        seasonal = score_product(product(product_id="s", occasions=("Новый год",)), context)
        plain = score_product(product(product_id="p"), context)
        self.assertGreater(seasonal.score, plain.score)

    def test_далёкий_повод_не_влияет(self):
        far = score_product(product(product_id="s", occasions=("Новый год",)), CONTEXT)
        plain = score_product(product(product_id="p"), CONTEXT)
        self.assertEqual(far.score, plain.score)

    def test_частые_публикации_опускают_товар(self):
        fresh = score_product(product(product_id="f", posts_30d=0), CONTEXT)
        tired = score_product(product(product_id="t", posts_30d=3), CONTEXT)
        self.assertGreater(fresh.score, tired.score)
        self.assertTrue(tired.risks)

    def test_при_равных_баллах_первым_идёт_более_прибыльный(self):
        a = product(product_id="a", title="Аааа", price_rub=4000, cost_rub=1800, posts_30d=2)
        b = product(product_id="b", title="Бббб", price_rub=500, cost_rub=390, posts_30d=0)
        ranked = rank_products([a, b], CONTEXT, limit=2)
        self.assertEqual([r.score for r in ranked][0], max(r.score for r in ranked))
        self.assertEqual(ranked[0].product_id, "a")

    def test_пустой_ответ_когда_продвигать_нечего(self):
        ranked = rank_products([product(stock=0), product(product_id="p2", stock=0)], CONTEXT)
        self.assertEqual(ranked, [])

    def test_лимит_соблюдается(self):
        products = [product(product_id=f"p{i}", title=f"Товар {i}") for i in range(10)]
        self.assertEqual(len(rank_products(products, CONTEXT, limit=3)), 3)

    def test_каждый_балл_объяснён(self):
        result = score_product(product(), CONTEXT)
        self.assertEqual(result.score, min(100, sum(f.points for f in result.factors)))


if __name__ == "__main__":
    unittest.main()
