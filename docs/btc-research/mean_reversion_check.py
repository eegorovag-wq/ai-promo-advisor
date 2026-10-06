"""Проверка возврата к среднему на BTCUSDT Spot: отзыв результата VR<1.

Детерминированный скрипт. Единственный источник случайности — np.random.default_rng(SEED),
поэтому вывод побайтно воспроизводим. Входные файлы хешируются по SHA-256,
хеши пишутся в отчёт — сверьте их со своими, скрипт сам ничего не сверяет.

Запуск:  python mean_reversion_check.py [путь-к-папке-с-данными]
         Без аргумента берётся $BTCUSDT_SPOT_6M, иначе ./DATA_BTCUSDT_SPOT_6M,
         иначе ../DATA_BTCUSDT_SPOT_6M.
Выход:   MEAN_REVERSION_CHECK.json  (машинный)
         MEAN_REVERSION_CHECK.md    (человеческий, см. render_report.py)

Данные — клайны Bybit Spot BTCUSDT, скачиваются публичным эндпоинтом
/v5/market/kline. Колонки: open_time_ms,open,high,low,close,volume.
Ожидаемые SHA-256 входных файлов указаны в готовом отчёте.
"""
from __future__ import annotations
import csv, hashlib, json, os, platform, sys
from pathlib import Path
import numpy as np

SEED = 20260928
NSIM = 1000
QS = [2, 4, 8, 16, 32, 64, 128, 256]
HERE = Path(__file__).resolve().parent


def resolve_data_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    env = os.environ.get("BTCUSDT_SPOT_6M")
    if env:
        return Path(env).expanduser().resolve()
    for cand in (HERE / "DATA_BTCUSDT_SPOT_6M", HERE.parent / "DATA_BTCUSDT_SPOT_6M"):
        if cand.is_dir():
            return cand
    return HERE / "DATA_BTCUSDT_SPOT_6M"


DATA = resolve_data_dir()
FILES = {
    "1m":  "BTCUSDT_SPOT_1_2026-03-01_2026-09-01.csv",
    "5m":  "BTCUSDT_SPOT_5_2026-03-01_2026-09-01.csv",
    "15m": "BTCUSDT_SPOT_15_2026-03-01_2026-09-01.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_closes(path: Path) -> np.ndarray:
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(float(row["close"]))
    return np.asarray(out, dtype=float)


def vr(r: np.ndarray, q: int) -> float:
    """Lo-MacKinlay variance ratio, несмещённая оценка."""
    n = r.size
    mu = r.mean()
    d = r - mu
    sa = (d @ d) / (n - 1)
    cs = np.concatenate(([0.0], np.cumsum(r)))
    agg = cs[q:] - cs[:-q]
    m = q * (n - q + 1) * (1 - q / n)
    return (((agg - q * mu) ** 2).sum() / m) / sa


def vr_z(r: np.ndarray, q: int) -> tuple[float, float, float]:
    """Возвращает (VR, z гомоскедастичный M1, z робастный M2)."""
    n = r.size
    mu = r.mean()
    d = r - mu
    v = vr(r, q)
    phi1 = 2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * n)
    z1 = (v - 1.0) / np.sqrt(phi1)
    d2 = d ** 2
    den = d2.sum() ** 2
    phi2 = 0.0
    for j in range(1, q):
        phi2 += (2.0 * (q - j) / q) ** 2 * ((d2[j:] @ d2[:-j]) / den)
    return v, z1, (v - 1.0) / np.sqrt(phi2)


def main() -> int:
    if not DATA.is_dir():
        print("Папка с данными не найдена: %s" % DATA, file=sys.stderr)
        print("Укажите её первым аргументом или в $BTCUSDT_SPOT_6M.", file=sys.stderr)
        return 2
    missing = [f for f in FILES.values() if not (DATA / f).exists()]
    if missing:
        print("В %s не хватает файлов: %s" % (DATA, ", ".join(missing)), file=sys.stderr)
        return 2
    print("данные: %s" % DATA)

    rng = np.random.default_rng(SEED)
    report: dict = {
        "seed": SEED,
        "surrogate_draws": NSIM,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "inputs": {tf: {"file": f, "sha256": sha256(DATA / f)} for tf, f in FILES.items()},
        "timeframes": {},
    }

    for tf, fname in FILES.items():
        close = load_closes(DATA / fname)
        r = np.diff(np.log(close))
        ek = float(((r - r.mean()) ** 4).mean() / r.var() ** 2 - 3.0)
        block = {
            "bars": int(r.size),
            "excess_kurtosis": round(ek, 2),
            "buy_and_hold_pct": round(float(close[-1] / close[0] - 1.0) * 100, 2),
            "q": {},
        }
        for q in QS:
            v, z1, z2 = vr_z(r, q)
            shuf = np.array([vr(rng.permutation(r), q) for _ in range(NSIM)])
            sign = np.array([vr(r * rng.choice([-1.0, 1.0], r.size), q) for _ in range(NSIM)])
            s_lo, s_hi = (float(x) for x in np.percentile(shuf, [2.5, 97.5]))
            g_lo, g_hi = (float(x) for x in np.percentile(sign, [2.5, 97.5]))
            block["q"][str(q)] = {
                "vr": round(float(v), 4),
                "z_homoskedastic": round(float(z1), 2),
                "z_robust_M2": round(float(z2), 2),
                "shuffle_ci95": [round(s_lo, 4), round(s_hi, 4)],
                "shuffle_outside": bool(v < s_lo or v > s_hi),
                "signflip_ci95": [round(g_lo, 4), round(g_hi, 4)],
                "signflip_outside": bool(v < g_lo or v > g_hi),
                "signflip_pctile": round(float((sign < v).mean()) * 100, 1),
            }
        # устойчивость по подпериодам
        block["subperiods_q64"] = [
            {"block": i, "bars": int(b.size), "vr": round(float(vr_z(b, 64)[0]), 4),
             "z_robust_M2": round(float(vr_z(b, 64)[2]), 2)}
            for i, b in enumerate(np.array_split(r, 6), 1)
        ]
        report["timeframes"][tf] = block

    out_json = Path(__file__).resolve().parent / "MEAN_REVERSION_CHECK.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("записан %s" % out_json.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
