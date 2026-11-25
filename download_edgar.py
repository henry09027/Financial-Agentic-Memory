#!/usr/bin/env python3
"""Step 1a: download SEC EDGAR Company Facts for the shared company universe.

    python download_edgar.py --user-agent "Your Name email@example.com"
    python download_edgar.py --user-agent "..." --resume          # continue / retry failures
    python download_edgar.py --self-test

Input : artifacts/raw/universe.json (build_universe.py). One request per unique CIK
        (primary listing); every listed ticker of the CIK is kept in `tickers`.
Output: artifacts/raw/edgar_facts.json (fam-envelope-v1, kind=edgar_facts).
        Only the six configured metric families and their units are saved.

Record fields
  pipeline: cik, ticker (primary listing), tickers, entity_name (universe name, as in the broker
            stream; SEC's entityName if the universe has none), sec_entity_name, taxonomy,
            concept, item, label, unit
  SEC payload (unchanged names): start, end, val, accn, fy, fp, form, filed, frame
Pipeline-added fields are snake_case; provider fields keep SEC's names.

Behavior shared with download_broker.py: per-entity checkpoint/cache, --resume,
--overwrite protection, exponential backoff honoring Retry-After, outputs always
exported with metadata.status complete|partial. HTTP 404 (no XBRL facts) is a
recorded empty result; HTTP 401/403 aborts (fix the User-Agent).
Exit codes: 0 complete, 1 error, 2 partial (resume later), 130 interrupted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from importlib import metadata as importlib_metadata
import os
import re

from pipeline_common import (DEFAULT_PATHS, AbortDownload, PipelineError, RetryableFetchError, add_downloader_arguments,
                             get_logger, load_universe, normalize_cik, require, run_download, run_main,
                             self_test_download_driver, setup_logging, validate_downloader_args)

LOG = get_logger("download_edgar")
SOURCE_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ITEMS = {
    "revenue": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": "USD",
        "RevenueFromContractWithCustomerIncludingAssessedTax": "USD",
        "Revenues": "USD", "SalesRevenueNet": "USD"},
    "net_income": {"NetIncomeLoss": "USD"},
    "diluted_eps": {"EarningsPerShareDiluted": "USD/shares"},
    "operating_income": {"OperatingIncomeLoss": "USD"},
    "operating_cash_flow": {"NetCashProvidedByUsedInOperatingActivities": "USD"},
    "capital_expenditure": {"PaymentsToAcquirePropertyPlantAndEquipment": "USD"},
}
CONCEPTS = {concept: (item, unit) for item, concepts in ITEMS.items() for concept, unit in concepts.items()}


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            return (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError):
            return None


def records_for_company(data: dict, entity: dict, tickers: list[str]) -> list[dict]:
    cik = entity["cik"]
    if normalize_cik(data.get("cik")) != cik or not isinstance(data.get("facts"), dict):
        raise ValueError(f"Malformed or mismatched Company Facts response for {cik}")
    concepts = data["facts"].get("us-gaap", {})
    if not isinstance(concepts, dict):
        raise ValueError(f"Invalid us-gaap taxonomy for {cik}")
    records = []
    for concept, (item, unit) in CONCEPTS.items():
        if concept not in concepts:
            continue
        info = concepts[concept]
        if not isinstance(info, dict) or not isinstance(info.get("units"), dict):
            raise ValueError(f"Invalid units for {cik}/{concept}")
        observations = info["units"].get(unit, [])
        if not isinstance(observations, list) or not all(isinstance(row, dict) for row in observations):
            raise ValueError(f"Invalid observations for {cik}/{concept}")
        for observation in observations:
            # Native JSON numbers are kept; no float conversion.
            records.append(dict(observation, cik=cik, ticker=entity["ticker"], tickers=tickers,
                                entity_name=entity.get("name") or data.get("entityName"),
                                sec_entity_name=data.get("entityName"), taxonomy="us-gaap",
                                concept=concept, item=item, label=info.get("label"), unit=unit))
    return records


def make_fetcher(session, timeout: float, tickers_by_cik: dict[str, list[str]]):
    import requests

    def fetch(entity: dict, endpoint: str):
        def run() -> dict:
            url = SOURCE_URL.format(cik=entity["cik"])
            try:
                with session.get(url, timeout=(10, timeout)) as response:
                    status = response.status_code
                    if status == 200:
                        data = response.json()
                        if not isinstance(data, dict):
                            raise ValueError("Expected a JSON object")
                        records = records_for_company(data, entity, tickers_by_cik[entity["cik"]])
                        return {"status": "ok" if records else "empty", "records": records,
                                "empty_reason": None if records else "No configured concepts/units reported"}
                    if status == 404:
                        return {"status": "empty", "records": [], "empty_reason": "SEC returned HTTP 404 (no Company Facts)"}
                    if status in (401, 403):
                        raise AbortDownload(f"HTTP {status} from SEC; check --user-agent and SEC access policy")
                    if status in (429, 500, 502, 503, 504):
                        raise RetryableFetchError(f"HTTP {status}", retry_after_seconds(response.headers.get("Retry-After")))
                    raise RuntimeError(f"HTTP {status}: {url}")
            except requests.RequestException as error:
                raise RetryableFetchError(str(error)) from error
        return run
    return fetch


def self_test() -> None:
    entity = {"cik": "0000320193", "ticker": "AAPL", "name": "Apple"}
    data = {"cik": 320193, "entityName": "Apple Inc.", "facts": {"us-gaap": {
        "Revenues": {"label": "Revenues", "units": {"USD": [
            {"start": "2020-01-01", "end": "2020-12-31", "val": 10, "accn": "0000320193-21-000001",
             "fy": 2020, "fp": "FY", "form": "10-K", "filed": "2021-02-01"}], "EUR": [{"val": 1}]}},
        "EarningsPerShareDiluted": {"label": "EPS", "units": {"USD/shares": [{"val": 1.5}]}},
        "Unrelated": {"units": {"USD": [{"val": 3}]}}}}}
    records = records_for_company(data, entity, ["AAPL"])
    require(len(records) == 2 and records[0]["item"] == "revenue" and records[0]["val"] == 10, "Concept filter failed")
    require(records[0]["entity_name"] == "Apple" and records[0]["sec_entity_name"] == "Apple Inc."
            and records[0]["ticker"] == "AAPL", "Identity fields failed")
    try:
        records_for_company(dict(data, cik=1), entity, ["AAPL"])
    except ValueError:
        pass
    else:
        raise PipelineError("Mismatched CIK accepted")
    require(retry_after_seconds("5") == 5 and retry_after_seconds("junk") is None, "Retry-After parsing failed")
    self_test_download_driver(LOG)
    LOG.info("EDGAR downloader self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT"),
                        help="'Name email@example.com' as SEC requires (env SEC_USER_AGENT)")
    add_downloader_arguments(parser, output_default=DEFAULT_PATHS["edgar_facts"], delay_default=0.2)
    args = parser.parse_args()
    setup_logging(args.log_level)
    if args.self_test:
        self_test()
        return 0
    validate_downloader_args(parser, args, minimum_delay=0.11)  # SEC fair access: at most 10 requests/second
    if not args.user_agent or not re.search(r"\S+@\S+\.\S+", args.user_agent) or re.search(r"[\r\n]", args.user_agent):
        parser.error("Set --user-agent or SEC_USER_AGENT to your name/organization and contact email")
    import requests
    listings, universe_reference = load_universe(args.universe)
    tickers_by_cik: dict[str, list[str]] = {}
    for listing in listings:
        tickers_by_cik.setdefault(listing["cik"], []).append(listing["ticker"])
    entities = [listing for listing in listings if listing["primary_listing"]]
    require(len(entities) == len(tickers_by_cik), "Universe must have exactly one primary listing per CIK")
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent, "Accept": "application/json"})
    base_metadata = {"source": "SEC EDGAR Company Facts API", "source_url_template": SOURCE_URL, "metrics": ITEMS,
                     "client_versions": {"requests": importlib_metadata.version("requests")},
                     "field_policy": "pipeline fields snake_case; SEC payload fields keep SEC names",
                     "coverage_note": "Complete means every request succeeded; it does not imply complete XBRL history."}
    try:
        return run_download(script=__file__, args=args, logger=LOG, entities=entities, entity_key="cik",
                            universe_reference=universe_reference, endpoints={"facts": (args.output, "edgar_facts")},
                            fetch=make_fetcher(session, args.timeout, tickers_by_cik), base_metadata=base_metadata)
    finally:
        session.close()


if __name__ == "__main__":
    run_main(main, LOG)
