"""Командная строка: весь рабочий цикл без интерфейса.

    python -m promo_advisor recommend
    python -m promo_advisor draft p04 --channel telegram --audience "постоянные клиенты"
    python -m promo_advisor queue
    python -m promo_advisor approve 1 --actor "Елена"
    python -m promo_advisor publish 1 --actor "Елена"
    python -m promo_advisor evaluate
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

from .catalog import load_context, load_products
from .evaluate import compare_versions
from .guardrails import GuardrailError, PublishPolicy
from .llm import get_provider
from .scoring import rank_products
from .workflow import PostStore, make_draft


def _print_recommendations(limit: int, today: date | None) -> None:
    products = load_products()
    context = load_context(today)
    for rec in rank_products(products, context, limit):
        print(f"\n{rec.score:>3}/100  {rec.title}  [{rec.product_id}]")
        for factor in rec.factors:
            print(f"        +{factor.points:<3} {factor.label}")
        for risk in rec.risks:
            print(f"        ! {risk}")
    print("\nЭто приоритет для проверки человеком, а не прогноз спроса и не ожидаемая прибыль.")


def _cmd_draft(args: argparse.Namespace) -> int:
    products = load_products()
    context = load_context(args.today)
    ranked = rank_products(products, context, limit=50)
    chosen = next((rec for rec in ranked if rec.product_id == args.product_id), None)
    if chosen is None:
        print(f"Товар {args.product_id} не найден среди пригодных к продвижению")
        return 1
    draft = make_draft(get_provider(), chosen, args.channel, args.audience)
    store = PostStore(args.db)
    post = store.create_draft(
        recommendation=chosen, draft=draft, channel=args.channel, actor=args.actor, discount_percent=args.discount
    )
    store.close()
    print(f"Черновик №{post.id} создан, статус «{post.status}».\n")
    print(post.text)
    print(f"\nПочему этот товар: {post.reason}")
    return 0


def _cmd_queue(args: argparse.Namespace) -> int:
    store = PostStore(args.db)
    posts = store.queue(args.status)
    if not posts:
        print("Очередь пуста")
    for post in posts:
        approved = f", согласовал: {post.approved_by}" if post.approved_by else ""
        print(f"№{post.id} [{post.status}] {post.title} — {post.channel}{approved}")
    store.close()
    return 0


def _cmd_approve(args: argparse.Namespace) -> int:
    store = PostStore(args.db)
    try:
        post = store.approve(args.post_id, args.actor)
        print(f"Публикация №{post.id} согласована: {post.approved_by}")
        return 0
    except (GuardrailError, LookupError) as error:
        print(f"Отказ: {error}")
        return 1
    finally:
        store.close()


def _cmd_publish(args: argparse.Namespace) -> int:
    store = PostStore(args.db)
    policy = PublishPolicy(kill_switch=args.kill_switch)
    moment = datetime.fromisoformat(args.at) if args.at else datetime.now()
    try:
        post = store.publish(args.post_id, policy, args.actor, now=moment)
        print(f"Публикация №{post.id} отправлена в {post.channel} ({post.published_at})")
        return 0
    except (GuardrailError, LookupError) as error:
        print(f"Отказ: {error}")
        return 1
    finally:
        store.close()


def _cmd_evaluate(_: argparse.Namespace) -> int:
    reports = compare_versions()
    for report in reports.values():
        print(f"\n{report.version}: {report.passed} из {report.total} ({report.accuracy:.0%})")
        for failure in report.failures():
            print(f"   не сошлось: {failure.scenario_id} — ожидали {failure.expected}, получили {failure.got}")
    print("\nСценарии размечены автором правил, это проверка логики, а не независимый замер спроса.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="promo_advisor", description="Советчик по продвижению товаров")
    parser.add_argument("--db", default="promo.db", help="файл базы SQLite")
    sub = parser.add_subparsers(dest="command", required=True)

    p_rec = sub.add_parser("recommend", help="показать приоритет продвижения")
    p_rec.add_argument("--limit", type=int, default=3)
    p_rec.add_argument("--today", type=date.fromisoformat, default=None)

    p_draft = sub.add_parser("draft", help="сделать черновик публикации")
    p_draft.add_argument("product_id")
    p_draft.add_argument("--channel", default="telegram", choices=("telegram", "max"))
    p_draft.add_argument("--audience", default="постоянные клиенты")
    p_draft.add_argument("--actor", default="менеджер")
    p_draft.add_argument("--discount", type=int, default=0)
    p_draft.add_argument("--today", type=date.fromisoformat, default=None)

    p_queue = sub.add_parser("queue", help="очередь публикаций")
    p_queue.add_argument("--status", default=None, choices=["draft", "approved", "rejected", "published"])

    p_approve = sub.add_parser("approve", help="согласовать публикацию")
    p_approve.add_argument("post_id", type=int)
    p_approve.add_argument("--actor", required=True)

    p_publish = sub.add_parser("publish", help="опубликовать согласованное")
    p_publish.add_argument("post_id", type=int)
    p_publish.add_argument("--actor", required=True)
    p_publish.add_argument("--at", default=None, help="время в формате ISO, для проверки тихих часов")
    p_publish.add_argument("--kill-switch", action="store_true", help="включить общий стоп")

    sub.add_parser("evaluate", help="внутренняя acceptance-проверка логики: v1 против v2")

    args = parser.parse_args(argv)
    if args.command == "recommend":
        _print_recommendations(args.limit, args.today)
        return 0
    if args.command == "draft":
        return _cmd_draft(args)
    if args.command == "queue":
        return _cmd_queue(args)
    if args.command == "approve":
        return _cmd_approve(args)
    if args.command == "publish":
        return _cmd_publish(args)
    return _cmd_evaluate(args)
