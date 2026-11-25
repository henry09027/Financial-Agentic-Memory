#!/usr/bin/env python3
"""Step 1b: download Yahoo Finance broker actions and stock splits for the shared universe.

    python download_broker.py
    python download_broker.py --resume                 # continue / retry failures
    python download_broker.py --self-test

Input  : artifacts/raw/universe.json (build_universe.py); every listing is requested
         by its Yahoo symbol (BRK.B -> BRK-B).
Outputs: artifacts/raw/broker_revisions.json (kind=broker_revisions)
         artifacts/raw/broker_splits.json    (kind=broker_splits; archived for provenance,
         not applied by the benchmark generator)

Record fields
  pipeline: ticker (listed, e.g. BRK.B), yahoo_symbol, cik, entity_name
  Yahoo payload (unchanged names): GradeDate (ISO-8601 UTC timestamp), Firm, ToGrade,
  FromGrade, Action, priceTargetAction, currentPriceTarget, priorPriceTarget; splits: date, ratio
Pipeline-added fields are snake_case; provider fields keep Yahoo's names.

Yahoo supplies structured actions, NOT broker research documents; coverage is what
Yahoo exposes. yfinance is an unofficial client: respect Yahoo's terms. yfinance is
pinned because empty-response detection depends on its exception text.
Behavior shared with download_edgar.py: per-entity checkpoint/cache, --resume,
--overwrite protection, exponential backoff, status complete|partial.
Exit codes: 0 complete, 1 error, 2 partial (resume later), 130 interrupted.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from importlib import metadata as importlib_metadata
import math
from pathlib import Path
from typing import Any

from pipeline_common import (DEFAULT_PATHS, PipelineError, add_downloader_arguments, get_logger, load_universe, require,
                             run_download, run_main, self_test_download_driver, setup_logging, validate_downloader_args)

LOG = get_logger("download_broker")
REVISION_FIELDS = ["GradeDate", "Firm", "ToGrade", "FromGrade", "Action",
                   "priceTargetAction", "currentPriceTarget", "priorPriceTarget"]


def json_value(value: Any) -> Any:
    import numpy as np
    import pandas as pd
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


def identity(entity: dict) -> dict:
    return {"ticker": entity["ticker"], "yahoo_symbol": entity["yahoo_symbol"], "cik": entity["cik"],
            "entity_name": entity.get("name")}


def revision_records(frame: Any, entity: dict) -> tuple[list[dict], list[str]]:
    import pandas as pd
    if frame is None:
        raise RuntimeError("Yahoo returned None rather than a revision table")
    if frame.empty:
        # yfinance returns an empty frame on swallowed HTTP failures; only its explicit
        # no-history exception (handled by the caller) counts as a verified empty result.
        raise RuntimeError("Unverified empty revision response; possible swallowed Yahoo HTTP error")
    frame = frame.reset_index()
    if "GradeDate" not in frame.columns:
        raise ValueError("Yahoo response has no GradeDate index/column")
    missing = [field for field in REVISION_FIELDS if field not in frame.columns]
    if missing:
        LOG.warning("%s: missing Yahoo fields %s; kept as null (rows without both price targets are later excluded by the generator)",
                    entity["yahoo_symbol"], missing)
    records = []
    for row in frame.to_dict("records"):
        timestamp = pd.to_datetime(row["GradeDate"], utc=True, errors="raise")
        if pd.isna(timestamp):
            raise ValueError("Yahoo revision date is missing")
        row["GradeDate"] = timestamp.isoformat()
        for field in missing:
            row[field] = None
        records.append(dict(json_value(row), **identity(entity)))
    return records, missing


def split_records(series: Any, entity: dict) -> list[dict]:
    if series is None:
        raise RuntimeError("Yahoo returned None rather than a splits series")
    rows = []
    for moment, ratio in series.items():
        ratio = float(ratio)
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("Invalid stock split ratio")
        rows.append(dict(identity(entity), date=moment.strftime("%Y-%m-%d"), ratio=ratio))
    return rows


def make_fetcher(timeout: float):
    def fetch(entity: dict, endpoint: str):
        def run() -> dict:
            import yfinance as yf
            from yfinance.exceptions import YFDataException
            symbol = entity["yahoo_symbol"]
            stock = yf.Ticker(symbol)
            if endpoint == "revisions":
                try:
                    frame = stock.get_upgrades_downgrades()
                except YFDataException as error:
                    if str(error) != f"No upgrade/downgrade history found for {symbol}":
                        raise
                    return {"status": "empty", "records": [], "missing_fields": [],
                            "empty_reason": "Yahoo explicitly returned no upgrade/downgrade history"}
                records, missing = revision_records(frame, entity)
                return {"status": "ok", "records": records, "missing_fields": missing, "empty_reason": None}
            history = stock.history(period="max", actions=True, auto_adjust=False, raise_errors=True, timeout=timeout)
            if history.empty or "Stock Splits" not in history.columns:
                raise ValueError("Yahoo returned no usable price history for stock splits")
            records = split_records(history.loc[history["Stock Splits"].ne(0), "Stock Splits"], entity)
            return {"status": "ok" if records else "empty", "records": records, "missing_fields": [],
                    "empty_reason": None if records else "No stock splits in price history"}
        return run
    return fetch


def self_test() -> None:
    import pandas as pd
    entity = {"ticker": "BRK.B", "yahoo_symbol": "BRK-B", "cik": "0001067983", "name": "Berkshire"}
    frame = pd.DataFrame({"GradeDate": pd.to_datetime(["2020-01-02 13:00"]), "Firm": ["A"], "ToGrade": ["Buy"],
                          "FromGrade": [""], "Action": ["up"], "priceTargetAction": ["Raises"],
                          "currentPriceTarget": [120.0], "priorPriceTarget": [float("nan")]}).set_index("GradeDate")
    records, missing = revision_records(frame, entity)
    require(not missing and records[0]["GradeDate"] == "2020-01-02T13:00:00+00:00", "GradeDate normalization failed")
    require(records[0]["priorPriceTarget"] is None and records[0]["ticker"] == "BRK.B" and records[0]["cik"] == entity["cik"],
            "Revision record fields failed")
    records, missing = revision_records(frame.drop(columns=["Action"]), entity)
    require(missing == ["Action"] and records[0]["Action"] is None, "Missing-field handling failed")
    try:
        revision_records(frame.iloc[0:0], entity)
    except RuntimeError:
        pass
    else:
        raise PipelineError("Unverified empty revision frame accepted")
    splits = split_records(pd.Series([2.0], index=pd.to_datetime(["2021-06-01"])), entity)
    require(splits == [dict(ticker="BRK.B", yahoo_symbol="BRK-B", cik=entity["cik"], entity_name="Berkshire",
                            date="2021-06-01", ratio=2.0)], "Split records failed")
    self_test_download_driver(LOG)
    LOG.info("Broker downloader self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits-output", type=Path, default=DEFAULT_PATHS["broker_splits"])
    add_downloader_arguments(parser, output_default=DEFAULT_PATHS["broker_revisions"], delay_default=1.0)
    args = parser.parse_args()
    setup_logging(args.log_level)
    if args.self_test:
        self_test()
        return 0
    validate_downloader_args(parser, args)
    require(args.output.resolve() != args.splits_output.resolve(), "--output and --splits-output must differ")
    entities, universe_reference = load_universe(args.universe)
    base_metadata = {"source": "Yahoo Finance via yfinance",
                     "client_versions": {name: importlib_metadata.version(name) for name in ("yfinance", "pandas")},
                     "field_policy": "pipeline fields snake_case; Yahoo payload fields keep Yahoo names",
                     "coverage_note": "Complete means every request succeeded; it does not imply complete Yahoo history."}
    return run_download(script=__file__, args=args, logger=LOG, entities=entities, entity_key="ticker",
                        universe_reference=universe_reference,
                        endpoints={"revisions": (args.output, "broker_revisions"),
                                   "splits": (args.splits_output, "broker_splits")},
                        fetch=make_fetcher(args.timeout), base_metadata=base_metadata)


if __name__ == "__main__":
    run_main(main, LOG)
