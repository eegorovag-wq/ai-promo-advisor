"""Путь публикации: черновик → согласование человеком → публикация.

Состояние и журнал действий лежат в SQLite (стандартная библиотека). Журнал —
не украшение: по нему видно, кто согласовал и что именно было опубликовано,
а при разборе жалобы это единственный источник правды.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .guardrails import GuardrailError, PublishPolicy, check_can_publish
from .llm import Draft, DraftRequest, LlmProvider
from .models import Recommendation

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT NOT NULL,
    title TEXT NOT NULL,
    channel TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft', 'approved', 'rejected', 'published')),
    text TEXT NOT NULL,
    reason TEXT NOT NULL,
    discount_percent INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    approved_by TEXT,
    approved_at TEXT,
    published_at TEXT,
    rejected_reason TEXT
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER NOT NULL REFERENCES posts(id),
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    at TEXT NOT NULL,
    details TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Post:
    id: int
    product_id: str
    title: str
    channel: str
    status: str
    text: str
    reason: str
    discount_percent: int
    approved_by: str | None = None
    published_at: str | None = None
    rejected_reason: str | None = None


class PostStore:
    def __init__(self, path: str | Path = "promo.db") -> None:
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        self.connection.close()

    def _audit(self, post_id: int, action: str, actor: str, details: dict[str, Any], at: datetime) -> None:
        self.connection.execute(
            "INSERT INTO audit(post_id, action, actor, at, details) VALUES (?,?,?,?,?)",
            (post_id, action, actor, at.isoformat(timespec="seconds"), json.dumps(details, ensure_ascii=False)),
        )

    def _row_to_post(self, row: sqlite3.Row) -> Post:
        return Post(
            id=row["id"],
            product_id=row["product_id"],
            title=row["title"],
            channel=row["channel"],
            status=row["status"],
            text=row["text"],
            reason=row["reason"],
            discount_percent=row["discount_percent"],
            approved_by=row["approved_by"],
            published_at=row["published_at"],
            rejected_reason=row["rejected_reason"],
        )

    def get(self, post_id: int) -> Post:
        row = self.connection.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone()
        if row is None:
            raise LookupError(f"Публикация {post_id} не найдена")
        return self._row_to_post(row)

    def queue(self, status: str | None = None) -> list[Post]:
        if status:
            rows = self.connection.execute("SELECT * FROM posts WHERE status=? ORDER BY id", (status,)).fetchall()
        else:
            rows = self.connection.execute("SELECT * FROM posts ORDER BY id").fetchall()
        return [self._row_to_post(row) for row in rows]

    def audit_trail(self, post_id: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT action, actor, at, details FROM audit WHERE post_id=? ORDER BY id", (post_id,)
        ).fetchall()
        return [
            {"action": r["action"], "actor": r["actor"], "at": r["at"], "details": json.loads(r["details"])}
            for r in rows
        ]

    def create_draft(
        self,
        *,
        recommendation: Recommendation,
        draft: Draft,
        channel: str,
        actor: str,
        discount_percent: int = 0,
        now: datetime | None = None,
    ) -> Post:
        moment = now or datetime.now()
        cursor = self.connection.execute(
            """INSERT INTO posts(product_id, title, channel, status, text, reason, discount_percent, created_at)
               VALUES (?,?,?,'draft',?,?,?,?)""",
            (
                recommendation.product_id,
                recommendation.title,
                channel,
                draft.full_text,
                draft.reason,
                discount_percent,
                moment.isoformat(timespec="seconds"),
            ),
        )
        post_id = int(cursor.lastrowid)
        self._audit(post_id, "draft_created", actor, {"score": recommendation.score}, moment)
        self.connection.commit()
        return self.get(post_id)

    def approve(self, post_id: int, actor: str, now: datetime | None = None) -> Post:
        moment = now or datetime.now()
        post = self.get(post_id)
        if post.status != "draft":
            raise GuardrailError(f"Согласовать можно только черновик, а статус «{post.status}»")
        if not actor:
            raise GuardrailError("Нужно указать, кто согласовал")
        self.connection.execute(
            "UPDATE posts SET status='approved', approved_by=?, approved_at=? WHERE id=?",
            (actor, moment.isoformat(timespec="seconds"), post_id),
        )
        self._audit(post_id, "approved", actor, {}, moment)
        self.connection.commit()
        return self.get(post_id)

    def reject(self, post_id: int, actor: str, reason: str, now: datetime | None = None) -> Post:
        moment = now or datetime.now()
        post = self.get(post_id)
        if post.status not in ("draft", "approved"):
            raise GuardrailError(f"Отклонить можно черновик или согласованное, а статус «{post.status}»")
        if not reason.strip():
            raise GuardrailError("Причина отклонения обязательна: по ней потом правят промпт")
        self.connection.execute(
            "UPDATE posts SET status='rejected', rejected_reason=? WHERE id=?", (reason.strip(), post_id)
        )
        self._audit(post_id, "rejected", actor, {"reason": reason.strip()}, moment)
        self.connection.commit()
        return self.get(post_id)

    def publish(self, post_id: int, policy: PublishPolicy, actor: str, now: datetime | None = None) -> Post:
        """Публикация проходит только через проверки. Повторная публикация запрещена."""
        moment = now or datetime.now()
        post = self.get(post_id)
        if post.status == "published":
            raise GuardrailError("Эта публикация уже отправлена")
        if post.status == "rejected":
            raise GuardrailError("Отклонённую публикацию отправить нельзя")
        check_can_publish(
            approved_by=post.approved_by,
            text=post.text,
            discount_percent=post.discount_percent,
            moment=moment,
            policy=policy,
        )
        self.connection.execute(
            "UPDATE posts SET status='published', published_at=? WHERE id=?",
            (moment.isoformat(timespec="seconds"), post_id),
        )
        self._audit(post_id, "published", actor, {"channel": post.channel}, moment)
        self.connection.commit()
        return self.get(post_id)


def make_draft(provider: LlmProvider, recommendation: Recommendation, channel: str, audience: str) -> Draft:
    return provider.generate_draft(DraftRequest(recommendation=recommendation, channel=channel, audience=audience))
