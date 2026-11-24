#!/usr/bin/env python3
"""Download Yahoo broker actions and stock splits. Python 3.10+ on Linux.

Setup:
    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -r requirements.txt
    python download_broker_evidence.py --output-dir data
    python download_broker_evidence.py --output-dir data --resume

Default universe: Wikipedia's current S&P 500 constituents, not historical
membership. If Wikipedia is blocked, use --tickers-file (one verified ticker per
line, # comments allowed). Custom Yahoo exchange suffixes are preserved; BRK.B
and BF.B are normalized to Yahoo share-class symbols.
Yahoo supplies structured actions, NOT original broker research documents.
Coverage is limited to what Yahoo exposes. A complete status means requests
succeeded, not that all historical broker actions exist. Respect Yahoo's terms
and data licensing; yfinance is an unofficial client.

Outputs: broker_revisions.json and broker_splits.json, each containing metadata
and records. Keep .broker_checkpoint.json and .broker_cache/ to resume; each
endpoint is checkpointed separately to avoid rewriting the full history.
Exit 2: failed endpoints; partial outputs saved. Exit 130: interrupted; resume.
No Snowflake credentials, services or notebook runtime are required.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

LOG = logging.getLogger("broker_download")
UNIVERSE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
REVISION_FIELDS = ["GradeDate", "Firm", "ToGrade", "FromGrade", "Action",
                   "priceTargetAction", "currentPriceTarget", "priorPriceTarget"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def normalize_tickers(values: list[str]) -> list[str]:
    aliases = {"BRK.B": "BRK-B", "BF.B": "BF-B"}
    symbols = [str(value).strip().upper() for value in values if str(value).strip()]
    tickers = sorted({aliases.get(symbol, symbol) for symbol in symbols})
    if not tickers or any(not re.fullmatch(r"[A-Z0-9][A-Z0-9.^=-]{0,19}", ticker) for ticker in tickers):
        raise ValueError("Ticker list is empty or contains invalid symbols")
    return tickers


def load_universe(tickers_file: Path | None, attempts: int, pause: float) -> tuple[list[str], dict]:
    if tickers_file:
        text = tickers_file.read_text(encoding="utf-8")
        values = [line.split("#", 1)[0].strip() for line in text.splitlines()]
        return normalize_tickers(values), {"source": str(tickers_file), "retrieved_at": utc_now(),
                                           "file_sha256": hashlib.sha256(text.encode()).hexdigest()}
    import pandas as pd
    import requests
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(UNIVERSE_URL, timeout=30,
                                    headers={"User-Agent": "Mozilla/5.0 (compatible; BrokerEvidenceDownloader/1.0)"})
            response.raise_for_status()
            tables = pd.read_html(io.StringIO(response.text))
            matches = [table for table in tables if {"Symbol", "Security", "GICS Sector"} <= set(table.columns)]
            if len(matches) != 1:
                raise ValueError("Could not uniquely identify the S&P 500 constituent table; use --tickers-file")
            tickers = normalize_tickers(matches[0]["Symbol"].tolist())
            if not 450 <= len(tickers) <= 550:
                raise ValueError("Unexpected constituent count; use an independently verified --tickers-file")
            return tickers, {"source": UNIVERSE_URL, "retrieved_at": utc_now(),
                             "membership": "current at retrieval; not historical membership"}
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(max(pause, 2) * (attempt + 1))
    raise RuntimeError(f"Constituent lookup failed: {last_error}. Supply --tickers-file.")


def json_value(value: Any) -> Any:
    import pandas as pd
    import numpy as np
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported Yahoo field type: {type(value).__name__}")


def revision_records(frame: Any, ticker: str) -> tuple[list[dict], list[str]]:
    if frame is None:
        raise RuntimeError("Yahoo returned None rather than a revision table")
    if frame.empty:
        # yfinance 0.2.66 returns an empty frame on swallowed HTTP failures.
        # Only its explicit no-history exception is accepted as a verified empty.
        raise RuntimeError("Unverified empty revision response; possible swallowed Yahoo HTTP error")
    frame = frame.reset_index()
    if "GradeDate" not in frame.columns:
        raise ValueError("Yahoo response has no GradeDate index/column")
    missing = [field for field in REVISION_FIELDS if field not in frame.columns]
    if missing:
        LOG.warning("%s: missing Yahoo fields %s; retained as null, may be excluded from benchmark", ticker, missing)
    records = []
    import pandas as pd
    for row in frame.to_dict("records"):
        timestamp = pd.to_datetime(row["GradeDate"], utc=True, errors="raise")
        if pd.isna(timestamp):
            raise ValueError("Yahoo revision date is missing")
        row["GradeDate"] = timestamp.isoformat()
        row["ticker"] = ticker
        for field in missing:
            row[field] = None
        records.append(json_value(row))
    return records, missing


def split_records(series: Any, ticker: str) -> list[dict]:
    if series is None:
        raise RuntimeError("Yahoo returned None rather than a splits series")
    rows = []
    for moment, ratio in series.items():
        ratio = float(ratio)
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("Invalid stock split ratio")
        rows.append({"ticker": ticker, "date": moment.strftime("%Y-%m-%d"), "ratio": ratio})
    return rows


def fetch_endpoint(ticker: str, endpoint: str, attempts: int, pause: float, timeout: float) -> dict:
    import yfinance as yf
    from yfinance.exceptions import YFDataException
    for attempt in range(attempts):
        try:
            stock = yf.Ticker(ticker)
            if endpoint == "revisions":
                try:
                    frame = stock.get_upgrades_downgrades()
                except YFDataException as error:
                    if str(error) != f"No upgrade/downgrade history found for {ticker}":
                        raise
                    return {"status": "empty", "retrieved_at": utc_now(), "records": [],
                            "missing_fields": [], "error": None, "empty_reason": "Yahoo explicitly returned no upgrade/downgrade history"}
                records, missing = revision_records(frame, ticker)
            else:
                history = stock.history(period="max", actions=True, auto_adjust=False,
                                        raise_errors=True, timeout=timeout)
                if history.empty or "Stock Splits" not in history.columns:
                    raise ValueError("Yahoo returned no usable price history for stock splits")
                records = split_records(history.loc[history["Stock Splits"].ne(0), "Stock Splits"], ticker)
                missing = []
            return {"status": "ok" if records else "empty", "retrieved_at": utc_now(),
                    "records": records, "missing_fields": missing, "error": None}
        except Exception as error:
            LOG.warning("%s %s attempt %d/%d: %s", ticker, endpoint, attempt + 1, attempts, error)
            if attempt + 1 == attempts:
                return {"status": "failed", "retrieved_at": utc_now(), "records": [],
                        "missing_fields": [], "error": f"{type(error).__name__}: {error}"}
            time.sleep(max(pause, 2) * (attempt + 1))
    raise RuntimeError("Unreachable retry state")


def cache_path(output_dir: Path, state: dict, ticker: str, endpoint: str) -> Path:
    name = hashlib.sha256(ticker.encode()).hexdigest() + "_" + endpoint + ".json"
    return output_dir / ".broker_cache" / state["snapshot_id"] / name


def export_snapshot(state: dict, endpoint: str, path: Path) -> int:
    records, statuses = [], {}
    for ticker in state["tickers"]:
        result = state["results"].get(ticker, {}).get(endpoint)
        if result is None:
            statuses[ticker] = {"status": "pending"}
            continue
        records.extend(result["records"])
        statuses[ticker] = {key: value for key, value in result.items() if key != "records"}
        statuses[ticker]["record_count"] = len(result["records"])
    failed = sum(item["status"] in {"failed", "pending"} for item in statuses.values())
    atomic_json(path, {"format_version": "broker-evidence-v1", "kind": endpoint,
        "metadata": {"snapshot_id": state["snapshot_id"], "started_at": state["started_at"],
                     "exported_at": utc_now(), "source": "Yahoo Finance via yfinance",
                     "universe": state["universe"], "tickers": state["tickers"],
                     "yfinance_version": state["yfinance_version"], "status": "partial" if failed else "complete",
                     "record_count": len(records), "ticker_status": statuses,
                     "coverage_note": "Complete means requests succeeded, not guaranteed complete Yahoo historical coverage."},
        "records": records})
    LOG.info("Saved %s: %d records; %d failed/pending tickers", path, len(records), failed)
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--tickers-file", type=Path)
    parser.add_argument("--resume", action="store_true", help="Reuse successes and retry failed endpoints from checkpoint")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds between Yahoo endpoints (default: 1)")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30, help="Price-history timeout; revisions use yfinance's built-in request timeout")
    args = parser.parse_args(argv)
    if args.attempts < 1 or not math.isfinite(args.pause) or args.pause < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("attempts and timeout must be positive; pause nonnegative; timeout/pause finite")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import yfinance as yf
    checkpoint = args.output_dir / ".broker_checkpoint.json"
    if args.resume:
        if not checkpoint.exists():
            parser.error("--resume requires an existing .broker_checkpoint.json")
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("format_version") != "broker-checkpoint-v2":
            raise ValueError("Unsupported checkpoint format")
        if state["yfinance_version"] != yf.__version__:
            raise ValueError("yfinance version differs from checkpoint; reinstall original version or start a new output directory")
        if args.tickers_file:
            tickers, _ = load_universe(args.tickers_file, args.attempts, args.pause)
            if tickers != state["tickers"]:
                raise ValueError("Ticker list differs from checkpoint; use a new output directory")
    else:
        tickers, universe = load_universe(args.tickers_file, args.attempts, args.pause)
        started = utc_now()
        state = {"format_version": "broker-checkpoint-v2", "started_at": started,
                 "snapshot_id": hashlib.sha256(json.dumps([started, tickers]).encode()).hexdigest(),
                 "tickers": tickers, "universe": universe, "yfinance_version": yf.__version__}
        atomic_json(checkpoint, state)
    state["results"] = {}
    for ticker in state["tickers"]:
        results = state["results"].setdefault(ticker, {})
        for endpoint in ("revisions", "splits"):
            cached = cache_path(args.output_dir, state, ticker, endpoint)
            if cached.exists():
                results[endpoint] = json.loads(cached.read_text(encoding="utf-8"))
    interrupted = False
    try:
        for index, ticker in enumerate(state["tickers"], 1):
            results = state["results"][ticker]
            for endpoint in ("revisions", "splits"):
                if results.get(endpoint, {}).get("status") in {"ok", "empty"}:
                    continue
                result = fetch_endpoint(ticker, endpoint, args.attempts, args.pause, args.timeout)
                atomic_json(cache_path(args.output_dir, state, ticker, endpoint), result)
                results[endpoint] = result
                time.sleep(args.pause)
            LOG.info("[%d/%d] %s: revisions=%s splits=%s", index, len(state["tickers"]), ticker,
                     results["revisions"]["status"], results["splits"]["status"])
    except KeyboardInterrupt:
        LOG.warning("Interrupted; checkpoint saved. Re-run with --resume.")
        interrupted = True
    failures = export_snapshot(state, "revisions", args.output_dir / "broker_revisions.json")
    failures += export_snapshot(state, "splits", args.output_dir / "broker_splits.json")
    return 130 if interrupted else (2 if failures else 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        LOG.error("Download failed: %s", error)
        raise SystemExit(1)
