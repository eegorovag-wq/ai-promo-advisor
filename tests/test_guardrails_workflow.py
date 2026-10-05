"""Тесты ограничений и рабочего цикла.

Здесь проверяется главное обещание проекта: без человека ничего не публикуется,
а запрещённое не проходит ни при каких настройках промпта.
"""

import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from datetime import date, datetime
from pathlib import Path

from promo_advisor.guardrails import GuardrailError, PublishPolicy, check_can_publish, check_quiet_hours
from promo_advisor.llm import LlmError, StubLlm, parse_draft
from promo_advisor.models import Factor, Recommendation
from promo_advisor.workflow import PostStore, make_draft

DAYTIME = datetime(2026, 10, 5, 12, 0)
NIGHT = datetime(2026, 10, 5, 23, 30)
EARLY = datetime(2026, 10, 5, 7, 0)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RECOMMENDATION = Recommendation(
    product_id="p1",
    title="Набор «Тёплый день»",
    score=80,
    factors=(Factor("stock_ok", "Остаток 12 шт.", 18), Factor("margin_ok", "Маржа 45%", 16)),
    risks=(),
    evidence={"stock": 12, "margin_rub": 2000.0, "units_sold_30d": 2, "posts_30d": 0},
)


class GuardrailTest(unittest.TestCase):
    def test_без_согласования_публиковать_нельзя(self):
        with self.assertRaises(GuardrailError):
            check_can_publish(approved_by=None, text="текст", discount_percent=0, moment=DAYTIME, policy=PublishPolicy())

    def test_обязательное_согласование_нельзя_отключить_политикой(self):
        with self.assertRaises(TypeError):
            PublishPolicy(require_approval=False)

    def test_общий_стоп_сильнее_согласования(self):
        with self.assertRaises(GuardrailError) as error:
            check_can_publish(
                approved_by="Елена", text="текст", discount_percent=0, moment=DAYTIME,
                policy=PublishPolicy(kill_switch=True),
            )
        self.assertIn("стоп", str(error.exception).lower())

    def test_скидка_выше_разрешённой_отклоняется(self):
        with self.assertRaises(GuardrailError):
            check_can_publish(approved_by="Елена", text="текст", discount_percent=25, moment=DAYTIME, policy=PublishPolicy())

    def test_запрещённые_обещания_отклоняются(self):
        with self.assertRaises(GuardrailError):
            check_can_publish(
                approved_by="Елена", text="Успей купить, только сегодня!", discount_percent=0,
                moment=DAYTIME, policy=PublishPolicy(),
            )

    def test_тихие_часы_считаются_через_полночь(self):
        with self.assertRaises(GuardrailError):
            check_quiet_hours(NIGHT)
        with self.assertRaises(GuardrailError):
            check_quiet_hours(EARLY)
        check_quiet_hours(DAYTIME)  # днём исключения быть не должно


class LlmContractTest(unittest.TestCase):
    def test_не_json_отклоняется(self):
        with self.assertRaises(LlmError):
            parse_draft("Конечно! Вот ваш текст: ...")

    def test_нехватка_полей_отклоняется(self):
        with self.assertRaises(LlmError):
            parse_draft('{"headline": "Заголовок"}')

    def test_лишние_поля_отклоняются(self):
        with self.assertRaises(LlmError):
            parse_draft(
                '{"headline":"З","body":"Т","reason":"Р","call_to_action":"П","discount":50}'
            )

    def test_слишком_длинный_заголовок_отклоняется(self):
        payload = '{"headline": "%s", "body": "т", "reason": "р", "call_to_action": "п"}' % ("я" * 61)
        with self.assertRaises(LlmError):
            parse_draft(payload)

    def test_заглушка_даёт_валидный_черновик_без_сети(self):
        draft = make_draft(StubLlm(), RECOMMENDATION, "telegram", "постоянные клиенты")
        self.assertIn(RECOMMENDATION.title, draft.full_text)
        self.assertTrue(draft.reason)


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = PostStore(Path(self.dir.name) / "test.db")
        draft = make_draft(StubLlm(), RECOMMENDATION, "telegram", "постоянные клиенты")
        self.post = self.store.create_draft(recommendation=RECOMMENDATION, draft=draft, channel="telegram", actor="агент")

    def tearDown(self):
        self.store.close()
        self.dir.cleanup()

    def test_черновик_нельзя_опубликовать_без_согласования(self):
        with self.assertRaises(GuardrailError):
            self.store.publish(self.post.id, PublishPolicy(), "агент", now=DAYTIME)

    def test_полный_путь_черновик_согласование_публикация(self):
        self.store.approve(self.post.id, "Елена")
        published = self.store.publish(self.post.id, PublishPolicy(), "Елена", now=DAYTIME)
        self.assertEqual(published.status, "published")
        actions = [item["action"] for item in self.store.audit_trail(self.post.id)]
        self.assertEqual(actions, ["draft_created", "approved", "published"])

    def test_повторная_публикация_запрещена(self):
        self.store.approve(self.post.id, "Елена")
        self.store.publish(self.post.id, PublishPolicy(), "Елена", now=DAYTIME)
        with self.assertRaises(GuardrailError):
            self.store.publish(self.post.id, PublishPolicy(), "Елена", now=DAYTIME)

    def test_отклонённое_не_публикуется_и_требует_причину(self):
        with self.assertRaises(GuardrailError):
            self.store.reject(self.post.id, "Елена", "   ")
        self.store.reject(self.post.id, "Елена", "тон не наш")
        with self.assertRaises(GuardrailError):
            self.store.publish(self.post.id, PublishPolicy(), "Елена", now=DAYTIME)

    def test_в_журнале_видно_кто_согласовал(self):
        self.store.approve(self.post.id, "Елена")
        trail = self.store.audit_trail(self.post.id)
        self.assertEqual(trail[-1]["actor"], "Елена")


class EvaluationTest(unittest.TestCase):
    def test_вторая_версия_не_хуже_первой(self):
        from promo_advisor.evaluate import compare_versions

        reports = compare_versions()
        self.assertGreater(reports["v2"].accuracy, reports["v1"].accuracy)

    def test_известные_пробелы_зафиксированы(self):
        """Три сценария v2 не проходит осознанно: это документированные ограничения."""
        from promo_advisor.evaluate import compare_versions

        failures = {item.scenario_id for item in compare_versions()["v2"].failures()}
        self.assertEqual(failures, {"S13-поставка-завтра", "S14-сезон-против-маржи", "S23-сумма-прибыли-против-доли"})


class CliRegressionTest(unittest.TestCase):
    def test_черновик_создаётся_один_раз_при_windows_cp1251(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "promo.db"
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "cp1251"
            env.pop("PROMO_LLM_API_KEY", None)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "promo_advisor",
                    "--db",
                    str(db_path),
                    "draft",
                    "p01",
                    "--channel",
                    "telegram",
                ],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
            self.assertIn("Черновик №1 создан", result.stdout.decode("utf-8"))
            with closing(sqlite3.connect(db_path)) as connection:
                count = connection.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
            self.assertEqual(count, 1)

    def test_cli_не_предлагает_обход_согласования(self):
        result = subprocess.run(
            [sys.executable, "-m", "promo_advisor", "publish", "--help"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertNotIn(b"allow-without-approval", result.stdout)


if __name__ == "__main__":
    unittest.main()


class TextQualityTest(unittest.TestCase):
    """Текст проверяется правилами, а не на глаз: иначе две версии промпта не сравнить."""

    def _draft(self, **overrides):
        from promo_advisor.llm import Draft

        base = dict(
            headline="Набор «Тёплый день»",
            body="Набор «Тёплый день» — спокойный подарок на любой повод, соберём и упакуем.",
            reason="Выбран по данным магазина",
            call_to_action="Напишите нам, подберём подарок.",
        )
        base.update(overrides)
        return Draft(**base)

    def test_хороший_текст_проходит(self):
        from promo_advisor.text_quality import check_draft

        self.assertEqual(check_draft(self._draft(), RECOMMENDATION), [])

    def test_выдуманное_число_ловится(self):
        from promo_advisor.text_quality import check_draft

        issues = check_draft(self._draft(body="Набор «Тёплый день» всего за 999 рублей, подробности у нас."), RECOMMENDATION)
        self.assertIn("invented_number", {issue.code for issue in issues})

    def test_запрещённая_формулировка_ловится(self):
        from promo_advisor.text_quality import check_draft

        issues = check_draft(self._draft(body="Набор «Тёплый день» — только сегодня дешевле не найдёшь нигде."), RECOMMENDATION)
        self.assertIn("banned_phrase", {issue.code for issue in issues})

    def test_без_упоминания_товара_ловится(self):
        from promo_advisor.text_quality import check_draft

        issues = check_draft(self._draft(headline="Подарок", body="Отличное предложение для вашего праздника и близких."), RECOMMENDATION)
        self.assertIn("product_not_mentioned", {issue.code for issue in issues})

    def test_новый_промпт_чище_старого(self):
        from promo_advisor.evaluate import compare_prompts

        reports = compare_prompts()
        self.assertEqual(reports["v1"].clean, 0)
        self.assertEqual(reports["v2"].clean, reports["v2"].total)
