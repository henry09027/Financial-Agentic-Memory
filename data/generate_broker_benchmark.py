#!/usr/bin/env python3
"""Offline broker evidence retrieval benchmark, ported from download_broker.ipynb.

Requires Python 3.10+. Install: python -m pip install -r requirements.txt
Usage:
    python generate_broker_benchmark.py --self-test
    python generate_broker_benchmark.py --revisions data/broker_revisions.json
    python generate_broker_benchmark.py --revisions data/broker_revisions.json \
        --output benchmark_dataset_broker.json --rows-per-stratum 300

Default: 9,000 examples = 5 difficulties x 6 context sizes x 300 rows.
All ten original templates, positive-target prefilter, date/tie semantics,
fixed-radius neighborhoods, deterministic sampling and family splits are kept.
No LLM, API key, Yahoo connection, or Snowflake service is used by this script.
Stock splits are archived by the downloader but NOT applied to price targets.

Generation can take hours. Quotas may be impossible for some source snapshots.
Incomplete balance NEVER replaces an existing output. Candidate audits stream
to temporary disk rather than accumulating in RAM; allow ample local disk space.
The output embeds private gold and researcher metadata: send only question,
cutoff and evidence_list to a retriever, not the whole example.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import date, timedelta
from decimal import Decimal
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger("broker_benchmark")
CONFIG = {
    "seed": 20261124, "version": "dataset-v5-balanced-contexts",
    "sizes": [5, 10, 20, 30, 40, 50], "rows_per_stratum": 300,
    "max_candidates_per_level": 400000, "progress_every": 10000,
    "adjacency_days": [7, 14, 30, 60, 90, 180],
    "question_window_days": [30, 90, 180, 365],
    "dollar_thresholds": [5, 10, 20], "percent_thresholds": [5, 10, 20],
    "evidence_id_hex_chars": 12,
}
EVIDENCE_FIELDS = ["evidence_id", "date", "ticker", "firm", "currentPriceTarget", "priorPriceTarget"]
SOURCE_POLICY = {
    "name": "complete_positive_targets_v1",
    "required_nonempty": ["date", "ticker", "firm"],
    "required_finite_positive": ["currentPriceTarget", "priorPriceTarget"],
    "date_format": "YYYY-MM-DD", "timezone": "UTC",
    "deduplicate_on": ["date", "ticker", "firm", "currentPriceTarget", "priorPriceTarget"],
    "temporal_scope": "Latest/earliest refer to reports remaining after this prefilter.",
}
COMMON_INSTRUCTION = (
    "Return only a JSON list of the selected evidence_id values, each once. Do not return target values or a written answer. "
    "Match ticker and firm names exactly. All supplied reports have both targets available; latest and earliest refer to this filtered report collection."
)
TEMPLATES = {
    "L1_current": {"level": 1, "kind": "lookup", "field": "currentPriceTarget"},
    "L1_prior": {"level": 1, "kind": "lookup", "field": "priorPriceTarget"},
    "L2_raise_dollars": {"level": 2, "kind": "filter"},
    "L2_cut_percent": {"level": 2, "kind": "filter"},
    "L3_two_firms": {"level": 3, "kind": "latest"},
    "L3_two_tickers": {"level": 3, "kind": "latest"},
    "L4_boundary": {"level": 4, "kind": "boundary"},
    "L4_window": {"level": 4, "kind": "window"},
    "L5_boundary": {"level": 5, "kind": "boundary"},
    "L5_window": {"level": 5, "kind": "window"},
}


def require(condition: bool, message: str) -> None:
    # Explicit checks still run under python -O; notebook assert statements do not.
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def checked_short_id(full_hash: str, registry: dict, width: int = 12) -> str:
    require(1 <= width <= 64, "Invalid ID width")
    result = "e_" + full_hash[:width]
    require(result not in registry or registry[result] == full_hash, "Short ID collision; increase width and regenerate")
    registry[result] = full_hash
    return result


def prepare_evidence(source: pd.DataFrame, width: int = 12) -> tuple[list[dict], dict]:
    frame = source.copy(deep=True)
    frame.columns = [str(column).upper() for column in frame.columns]
    mapping = {"GRADEDATE": "date", "TICKER": "ticker", "FIRM": "firm",
               "CURRENTPRICETARGET": "currentPriceTarget", "PRIORPRICETARGET": "priorPriceTarget"}
    require(frame.columns.is_unique, "Ambiguous duplicate column names")
    missing = sorted(set(mapping) - set(frame.columns))
    require(not missing, f"Missing revision fields: {missing}. Check downloader missing_fields metadata.")
    frame = frame[list(mapping)].rename(columns=mapping)
    parsed = pd.to_datetime(frame.date, errors="coerce", utc=True, format="mixed")
    valid = parsed.notna() & frame.ticker.notna() & frame.firm.notna()
    valid &= frame.ticker.astype(str).str.strip().ne("") & frame.firm.astype(str).str.strip().ne("")
    invalid_count = int((~valid).sum())
    frame = frame.loc[valid].copy()
    frame["date"] = parsed.loc[valid].dt.strftime("%Y-%m-%d")
    for field in ["ticker", "firm"]:
        frame[field] = frame[field].astype(str).str.strip()
    target_valid = pd.Series(True, index=frame.index)
    unavailable = {}
    for field in ["currentPriceTarget", "priorPriceTarget"]:
        values = pd.to_numeric(frame[field], errors="coerce")
        usable_mask = values.gt(0) & np.isfinite(values)
        unavailable[field] = int((~usable_mask).sum())
        target_valid &= usable_mask
        frame[field] = values.astype(float)
    excluded_targets = int((~target_valid).sum())
    frame = frame.loc[target_valid].copy()
    records, registry, seen = [], {}, set()
    for record in frame.to_dict("records"):
        fingerprint = stable_hash(record)
        evidence_id = checked_short_id(fingerprint, registry, width)
        if fingerprint not in seen:
            records.append(dict(evidence_id=evidence_id, **record))
            seen.add(fingerprint)
    records.sort(key=lambda record: (record["date"], record["evidence_id"]))
    require(bool(records), "No complete-target evidence remains")
    return records, {
        "source_rows": len(source), "invalid_identity_or_date_rows": invalid_count,
        "excluded_target_rows": excluded_targets, "invalid_target_counts_overlapping": unavailable,
        "projected_duplicate_rows": len(frame) - len(records), "evidence_rows": len(records),
        "earliest": records[0]["date"], "latest": records[-1]["date"],
        "snapshot_hash": stable_hash(sorted(seen)), "source_policy": SOURCE_POLICY,
    }


def usable(value: Any) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def decimal_value(value: Any) -> Decimal:
    return Decimal(str(value))


def resolve_gold(records: list[dict], template_id: str, params: dict) -> set[str]:
    spec = TEMPLATES[template_id]
    rows = [record for record in records if record["ticker"] in params["tickers"] and record["date"] <= params["cutoff"]]

    def choose(candidates, order=None, field="currentPriceTarget"):
        if not candidates:
            raise ValueError("missing_endpoint")
        if order:
            selected_date = (min if order == "first" else max)(record["date"] for record in candidates)
            candidates = [record for record in candidates if record["date"] == selected_date]
        if len(candidates) != 1:
            raise ValueError("ambiguous_same_day")
        if not usable(candidates[0][field]):
            raise ValueError("unavailable_selected_target")
        return candidates[0]

    if spec["kind"] == "lookup":
        return {choose([record for record in rows if record["firm"] == params["firms"][0] and record["date"] == params["date"]], field=spec["field"])["evidence_id"]}
    if spec["kind"] == "filter":
        selected = set()
        for record in rows:
            if not params["start"] <= record["date"] <= params["end"]:
                continue
            current, prior = record["currentPriceTarget"], record["priorPriceTarget"]
            if not usable(current) or not usable(prior):
                continue
            current, prior, threshold = map(decimal_value, [current, prior, params["threshold"]])
            matches = current - prior >= threshold if template_id == "L2_raise_dollars" else 100 * (prior - current) >= threshold * prior
            if matches:
                selected.add(record["evidence_id"])
        return selected
    entities = ([(ticker, params["firms"][0]) for ticker in params["tickers"]] if template_id == "L3_two_tickers"
                else [(params["tickers"][0], firm) for firm in params["firms"]])
    selected, changes = set(), []
    for ticker, firm in entities:
        group = [record for record in rows if record["ticker"] == ticker and record["firm"] == firm]
        if spec["kind"] == "latest":
            selected.add(choose(group, "last")["evidence_id"])
            continue
        if spec["kind"] == "boundary":
            first = choose([record for record in group if record["date"] < params["boundary"]], "last")
            last = choose([record for record in group if record["date"] >= params["boundary"]], "first")
        else:
            window = [record for record in group if params["start"] <= record["date"] <= params["end"]]
            first, last = choose(window, "first"), choose(window, "last")
        if first["evidence_id"] == last["evidence_id"]:
            raise ValueError("endpoints_not_distinct")
        selected.update([first["evidence_id"], last["evidence_id"]])
        changes.append(decimal_value(last["currentPriceTarget"]) - decimal_value(first["currentPriceTarget"]))
    if spec["level"] == 5 and changes[0] == changes[1]:
        raise ValueError("tied_comparison")
    return selected


def predicate(field, operator, value):
    return {"field": field, "operator": operator, "value": value}


def golden_conditions(template_id: str, params: dict) -> dict:
    spec, branches = TEMPLATES[template_id], []

    def branch(name, ticker, firm, times, mode, field="currentPriceTarget", values=None):
        entities = [predicate("ticker", "=", ticker)]
        if firm is not None:
            entities.append(predicate("firm", "=", firm))
        branches.append({"name": name, "entity_filters": entities, "time_filters": times,
                         "value_filters": values or [], "selection": {"mode": mode, "order_by": "date",
                         "partition_by": ["ticker", "firm"], "required_target_field": field,
                         "tie_policy": "reject" if mode != "all_matches" else "include_all"},
                         "expected_rows": 1 if mode != "all_matches" else None})

    kind = spec["kind"]
    if kind == "lookup":
        branch("dated_report", params["tickers"][0], params["firms"][0], [predicate("date", "=", params["date"])], "exactly_one", spec["field"])
    elif kind == "filter":
        metric = "target_increase_dollars" if template_id == "L2_raise_dollars" else "target_decrease_percent"
        expression = "currentPriceTarget - priorPriceTarget" if metric == "target_increase_dollars" else "100 * (priorPriceTarget - currentPriceTarget) / priorPriceTarget"
        branch("matching_reports", params["tickers"][0], None, [predicate("date", ">=", params["start"]), predicate("date", "<=", params["end"])], "all_matches", values=[{"metric": metric, "expression": expression, "operator": ">=", "value": params["threshold"], "arithmetic": "decimal; percentage comparison uses cross multiplication"}])
    else:
        entities = ([(ticker, params["firms"][0]) for ticker in params["tickers"]] if template_id == "L3_two_tickers"
                    else [(params["tickers"][0], firm) for firm in params["firms"]])
        for index, (ticker, firm) in enumerate(entities):
            if kind == "latest":
                branch(f"entity_{index}_latest", ticker, firm, [], "latest")
            elif kind == "boundary":
                branch(f"entity_{index}_earlier", ticker, firm, [predicate("date", "<", params["boundary"])], "latest")
                branch(f"entity_{index}_later", ticker, firm, [predicate("date", ">=", params["boundary"])], "earliest")
            else:
                times = [predicate("date", ">=", params["start"]), predicate("date", "<=", params["end"])]
                branch(f"entity_{index}_earlier", ticker, firm, copy.deepcopy(times), "earliest")
                branch(f"entity_{index}_later", ticker, firm, copy.deepcopy(times), "latest")
    return {"version": "gold-selection-v1",
            "source_scope": {"collection": "prefiltered_reports", "policy": SOURCE_POLICY["name"],
                             "cutoff_filter": predicate("date", "<=", params["cutoff"])},
            "branches": branches, "combine": "UNION_DISTINCT",
            "validity_conditions": {"nonempty_gold": True, "all_branches_required": True,
                "distinct_endpoints": kind in ["boundary", "window"],
                "comparison": {"metric": "later_currentPriceTarget - earlier_currentPriceTarget",
                               "reject_equal_changes": spec["level"] == 5}},
            "expected_gold_count": None if kind == "filter" else len(branches)}


def condition_matches(record: dict, condition: dict) -> bool:
    value, expected = record[condition["field"]], condition["value"]
    operator = condition["operator"]
    if operator == "=":
        return value == expected
    if operator == "<":
        return value < expected
    if operator == "<=":
        return value <= expected
    if operator == ">=":
        return value >= expected
    raise ValueError("Unknown condition operator")


def evaluate_golden_conditions(records: list[dict], contract: dict) -> set[str]:
    # Independent evaluator: whitelist structured operations; never eval SQL/text.
    require(contract["version"] == "gold-selection-v1" and contract["combine"] == "UNION_DISTINCT", "Unsupported contract")
    eligible = [record for record in records if condition_matches(record, contract["source_scope"]["cutoff_filter"])]
    gold, endpoints = set(), defaultdict(list)
    for branch in contract["branches"]:
        conditions = branch["entity_filters"] + branch["time_filters"]
        chosen = [record for record in eligible if all(condition_matches(record, item) for item in conditions)]
        for value_filter in branch["value_filters"]:
            require(value_filter["operator"] == ">=", "Unsupported value operator")
            filtered = []
            for record in chosen:
                current, prior = record["currentPriceTarget"], record["priorPriceTarget"]
                if not usable(current) or not usable(prior):
                    continue
                current, prior, threshold = map(decimal_value, [current, prior, value_filter["value"]])
                if value_filter["metric"] == "target_increase_dollars":
                    matches = current - prior >= threshold
                elif value_filter["metric"] == "target_decrease_percent":
                    matches = (prior - current) * 100 >= threshold * prior
                else:
                    raise ValueError("Unsupported value metric")
                if matches:
                    filtered.append(record)
            chosen = filtered
        mode = branch["selection"]["mode"]
        if mode != "all_matches":
            if not chosen:
                raise ValueError("missing_endpoint")
            if mode in ["earliest", "latest"]:
                frame = pd.DataFrame(chosen)
                ranks = pd.to_datetime(frame.date).rank(method="min", ascending=mode == "earliest")
                chosen = frame.loc[ranks.eq(1)].to_dict("records")
            elif mode != "exactly_one":
                raise ValueError("Unsupported selection")
            if len(chosen) != 1:
                raise ValueError("ambiguous_same_day")
            if not usable(chosen[0][branch["selection"]["required_target_field"]]):
                raise ValueError("unavailable_selected_target")
        gold.update(record["evidence_id"] for record in chosen)
        if contract["validity_conditions"]["distinct_endpoints"]:
            for record in chosen:
                endpoints[(record["ticker"], record["firm"])].append(record)
    changes = []
    for reports in endpoints.values():
        if len(reports) != 2 or reports[0]["evidence_id"] == reports[1]["evidence_id"]:
            raise ValueError("endpoints_not_distinct")
        reports.sort(key=lambda record: record["date"])
        changes.append(decimal_value(reports[1]["currentPriceTarget"]) - decimal_value(reports[0]["currentPriceTarget"]))
    if contract["validity_conditions"]["comparison"]["reject_equal_changes"] and (len(changes) != 2 or changes[0] == changes[1]):
        raise ValueError("tied_comparison")
    expected = contract["expected_gold_count"]
    require(expected is None or len(gold) == expected, "incorrect_gold_count")
    return gold


def golden_conditions_sql(contract: dict) -> str:
    """Documentation over a logical relation, not an executed database query."""
    def literal(value):
        return "'" + value.replace("'", "''") + "'" if isinstance(value, str) else str(value)
    statements = []
    for branch in contract["branches"]:
        filters = [contract["source_scope"]["cutoff_filter"]] + branch["entity_filters"] + branch["time_filters"]
        terms = [f'{item["field"]} {item["operator"]} {literal(item["value"])}' for item in filters]
        terms.extend(f'({item["expression"]}) {item["operator"]} {item["value"]}' for item in branch["value_filters"])
        statement = "SELECT evidence_id FROM prefiltered_reports WHERE " + " AND ".join(terms)
        mode = branch["selection"]["mode"]
        if mode in ["earliest", "latest"]:
            direction = "ASC" if mode == "earliest" else "DESC"
            statement += f" QUALIFY DENSE_RANK() OVER (PARTITION BY ticker, firm ORDER BY date {direction}) = 1"
        if mode != "all_matches":
            statement += " /* REQUIRE EXACTLY ONE ROW; REJECT TIES */"
        statements.append("(" + statement + ")")
    return " UNION ".join(statements)


def private_answer(records: list[dict], template_id: str, params: dict, gold: set[str]) -> dict:
    chosen = sorted((record for record in records if record["evidence_id"] in gold), key=lambda record: (record["ticker"], record["firm"], record["date"]))
    kind = TEMPLATES[template_id]["kind"]
    if kind == "lookup":
        field = TEMPLATES[template_id]["field"]
        return {field: chosen[0][field]}
    if kind == "filter":
        return {"qualifying_report_count": len(gold)}
    if kind == "latest":
        return {"latest_targets": [{key: record[key] for key in ["ticker", "firm", "date", "currentPriceTarget"]} for record in chosen]}
    groups = defaultdict(list)
    for record in chosen:
        groups[record["firm"]].append(record)
    changes = {firm: float(decimal_value(reports[-1]["currentPriceTarget"]) - decimal_value(reports[0]["currentPriceTarget"])) for firm, reports in groups.items()}
    answer = {"signed_dollar_changes": changes}
    if TEMPLATES[template_id]["level"] == 5:
        answer["larger_signed_change_firm"] = max(changes, key=changes.get)
    return answer


def question_text(template_id: str, params: dict) -> str:
    ticker, firms, cutoff = params["tickers"][0], params["firms"], params["cutoff"]
    kind, level = TEMPLATES[template_id]["kind"], TEMPLATES[template_id]["level"]
    if kind == "lookup":
        label = "current" if template_id == "L1_current" else "prior"
        body = f'What {label} price target did {firms[0]} report for {ticker} on {params["date"]}? Select the one report matching this ticker, firm and date.'
    elif kind == "filter":
        change = (f'raised the target by at least ${params["threshold"]} (currentPriceTarget minus priorPriceTarget)' if template_id == "L2_raise_dollars"
                  else f'cut the target by at least {params["threshold"]}% (100 times priorPriceTarget minus currentPriceTarget, divided by priorPriceTarget)')
        body = f'Select every report for {ticker} dated from {params["start"]} to {params["end"]}, including both dates, that {change}.'
    elif template_id == "L3_two_firms":
        body = f'What are the latest current price targets for {ticker} from {firms[0]} and {firms[1]} as of {cutoff}? Select the most recent report dated on or before {cutoff} from each firm: two reports in total.'
    elif template_id == "L3_two_tickers":
        body = f'What are the latest current price targets from {firms[0]} for {params["tickers"][0]} and {params["tickers"][1]} as of {cutoff}? Select the most recent report dated on or before {cutoff} for each ticker: two reports in total.'
    else:
        subject = (f'For {ticker} and {firms[0]}, how did the current price target change?' if level == 4
                   else f'For {ticker}, which firm had the larger signed dollar change in current price target: {firms[0]} or {firms[1]}?')
        instruction = "Select these two reports" if level == 4 else "For EACH firm, select these two reports (four reports in total)"
        if kind == "boundary":
            selection = f'1. The most recent report dated before {params["boundary"]}. 2. The earliest report dated from {params["boundary"]} to {cutoff}, including both dates.'
        else:
            selection = f'1. The earliest report dated from {params["start"]} to {params["end"]}. 2. The latest report in that same date range. Include both boundary dates.'
        body = f'{subject} {instruction}: {selection} Compare the later target minus the earlier target.'
        if level == 5:
            body += " A smaller decrease counts as a larger signed change. Return reports for both firms, not only the winner."
    return body + " " + COMMON_INSTRUCTION


def natural_neighborhood(history: list[dict], gold: set[str], radius: int) -> list[dict]:
    gold_days = [date.fromisoformat(record["date"]).toordinal() for record in history if record["evidence_id"] in gold]
    return [record for record in history if any(abs(date.fromisoformat(record["date"]).toordinal() - day) <= radius for day in gold_days)]


def deterministic_filter(context, template_id, params, cutoff, gold, config):
    ids = [record["evidence_id"] for record in context]
    if len(ids) != len(set(ids)):
        return "duplicate_ids"
    if any(record["date"] > cutoff for record in context):
        return "future_evidence"
    if any(not usable(record[field]) for record in context for field in ["currentPriceTarget", "priorPriceTarget"]):
        return "incomplete_target_record"
    if not gold:
        return "no_gold"
    if not set(gold) <= set(ids):
        return "missing_gold"
    if not set(ids) - set(gold):
        return "no_adjacent_distractor"
    try:
        if resolve_gold(context, template_id, params) != set(gold):
            return "context_gold_mismatch"
    except ValueError as error:
        return str(error)
    if len(context) not in config["sizes"]:
        return "context_size_not_requested"
    return None


class ArrayHash:
    """Canonical SHA256 of a JSON array without materializing the array."""
    def __init__(self):
        self.digest = hashlib.sha256(b"[")
        self.count = 0

    def append(self, value):
        if self.count:
            self.digest.update(b",")
        self.digest.update(canonical(value))
        self.count += 1

    def hexdigest(self):
        result = self.digest.copy()
        result.update(b"]")
        return result.hexdigest()


class AttemptLog:
    def __init__(self, stream):
        self.stream = stream
        self.hash = ArrayHash()
        self.reasons = Counter()

    def append(self, audit):
        self.stream.write(canonical(audit) + b"\n")
        self.hash.append(audit)
        self.reasons[(audit["template_id"], audit["reason"])] += 1


class Benchmark:
    def __init__(self, evidence: list[dict], audit: dict, config: dict):
        self.evidence, self.audit, self.config = evidence, audit, config
        self.by_ticker, self.tickers_by_firm = defaultdict(list), defaultdict(set)
        for record in evidence:
            self.by_ticker[record["ticker"]].append(record)
            self.tickers_by_firm[record["firm"]].add(record["ticker"])
        self.tickers = sorted(self.by_ticker)

    def source_history(self, params):
        return [record for ticker in params["tickers"] for record in self.by_ticker[ticker] if record["date"] <= params["cutoff"]]

    def proposal(self, template_id, cutoff, rng):
        config = self.config
        ticker = rng.choice(self.tickers)
        rows = [record for record in self.by_ticker[ticker] if record["date"] <= cutoff]
        if not rows:
            raise ValueError("no_ticker_history")
        anchor = rng.choice(rows)
        groups = defaultdict(list)
        for record in rows:
            groups[record["firm"]].append(record)
        firms = sorted(groups)
        params = {"tickers": [ticker], "firms": [rng.choice(firms)], "cutoff": cutoff}
        spec = TEMPLATES[template_id]
        if spec["kind"] == "lookup":
            params.update(firms=[anchor["firm"]], date=anchor["date"])
        elif spec["kind"] == "filter":
            days = rng.choice(config["question_window_days"])
            params.update(start=(date.fromisoformat(anchor["date"]) - timedelta(days=days)).isoformat(), end=anchor["date"], threshold=rng.choice(config["dollar_thresholds"] if template_id == "L2_raise_dollars" else config["percent_thresholds"]))
        elif spec["kind"] == "latest":
            if template_id == "L3_two_tickers":
                firm = params["firms"][0]
                alternatives = [symbol for symbol in sorted(self.tickers_by_firm[firm]) if symbol != ticker and any(record["firm"] == firm and record["date"] <= cutoff for record in self.by_ticker[symbol])]
                if not alternatives:
                    raise ValueError("no_second_ticker")
                params["tickers"].append(rng.choice(alternatives))
            else:
                if len(firms) < 2:
                    raise ValueError("no_second_firm")
                params["firms"] = rng.sample(firms, 2)
        else:
            if spec["kind"] == "boundary":
                params["boundary"] = anchor["date"]
                eligible_firms = [firm for firm in firms if any(record["date"] < params["boundary"] for record in groups[firm]) and any(record["date"] >= params["boundary"] for record in groups[firm])]
            else:
                days = rng.choice(config["question_window_days"])
                params.update(start=(date.fromisoformat(anchor["date"]) - timedelta(days=days)).isoformat(), end=anchor["date"])
                eligible_firms = [firm for firm in firms if len({record["date"] for record in groups[firm] if params["start"] <= record["date"] <= params["end"]}) >= 2]
            needed = 2 if spec["level"] == 5 else 1
            if len(eligible_firms) < needed:
                raise ValueError("insufficient_temporal_coverage")
            params["firms"] = rng.sample(eligible_firms, needed)
        return params

    def generate(self, attempt_log: AttemptLog) -> list[dict]:
        config, examples = self.config, []
        first = date.fromisoformat(self.audit["earliest"]).toordinal()
        last = date.fromisoformat(self.audit["latest"]).toordinal()
        quota = config["rows_per_stratum"]
        for level in range(1, 6):
            templates = [name for name, spec in TEMPLATES.items() if spec["level"] == level]
            rngs = {name: random.Random(stable_hash([config["seed"], name])) for name in templates}
            counts, used_questions = Counter(), set()
            for attempt in range(config["max_candidates_per_level"]):
                if all(counts[size] == quota for size in config["sizes"]):
                    break
                if attempt and attempt % config["progress_every"] == 0:
                    LOG.info("L%d: %d candidates; counts %s", level, attempt, dict(sorted(counts.items())))
                template_id = templates[attempt % len(templates)]
                rng = rngs[template_id]
                cutoff = date.fromordinal(rng.randint(first, last)).isoformat()
                audit = dict(template_id=template_id, level=level, attempt=attempt, cutoff=cutoff, accepted=False, reason="", neighborhoods=[])
                try:
                    try:
                        params = self.proposal(template_id, cutoff, rng)
                        history = self.source_history(params)
                        gold = resolve_gold(history, template_id, params)
                        if not gold:
                            raise ValueError("no_gold")
                    except ValueError as error:
                        audit["reason"] = str(error)
                        continue
                    identity_params = {key: value for key, value in params.items() if key != "cutoff"}
                    identity_params["firms"] = sorted(identity_params["firms"])
                    identity_params["tickers"] = sorted(identity_params["tickers"])
                    signature = stable_hash([template_id, identity_params, sorted(gold)])
                    if signature in used_questions:
                        audit["reason"] = "duplicate_effective_question"
                        continue
                    used_questions.add(signature)
                    audit["source_gold_count"] = len(gold)
                    base_id = "b_" + stable_hash([config["version"], self.audit["snapshot_hash"], signature])[:24]
                    split_number = int(stable_hash(base_id)[:8], 16) % 10
                    split = "train" if split_number < 6 else ("dev" if split_number < 8 else "test")
                    seen_contexts, survivors = set(), []
                    for radius in config["adjacency_days"]:
                        rows = natural_neighborhood(history, gold, radius)
                        signature_context = tuple(sorted(record["evidence_id"] for record in rows))
                        item = dict(radius_days=radius, context_size=len(rows), accepted=False, reason="")
                        audit["neighborhoods"].append(item)
                        if signature_context in seen_contexts:
                            item["reason"] = "duplicate_neighborhood"
                            continue
                        seen_contexts.add(signature_context)
                        error = deterministic_filter(rows, template_id, params, cutoff, gold, config)
                        if error:
                            item["reason"] = error
                            continue
                        if counts[len(rows)] >= quota:
                            item["reason"] = "stratum_quota_full"
                            continue
                        item.update(accepted=True, reason="accepted")
                        survivors.append((radius, rows))
                    if not survivors:
                        audit["reason"] = "no_unfilled_valid_neighborhood"
                        continue
                    contract = golden_conditions(template_id, params)
                    require(evaluate_golden_conditions(history, contract) == gold, "Source gold resolvers disagree")
                    for radius, rows in survivors:
                        size = len(rows)
                        require(counts[size] < quota, "Quota exceeded")
                        require(evaluate_golden_conditions(rows, contract) == gold, "Context gold resolvers disagree")
                        shuffled = copy.deepcopy(rows)
                        random.Random(stable_hash([base_id, radius, config["seed"]])).shuffle(shuffled)
                        distractors = [record for record in rows if record["evidence_id"] not in gold]
                        examples.append(dict(
                            question=question_text(template_id, params), evidence_list=shuffled,
                            golden_evidence_ids=sorted(gold),
                            distractor_ids=[record["evidence_id"] for record in shuffled if record["evidence_id"] not in gold],
                            example_id=base_id + "_" + str(size), base_id=base_id, context_size=size,
                            gold_count=len(gold), difficulty=level, template_id=template_id, cutoff=cutoff, split=split,
                            gold_fraction=len(gold) / size, parameters=copy.deepcopy(params),
                            golden_conditions=copy.deepcopy(contract), golden_conditions_sql_like=golden_conditions_sql(contract),
                            adjacency_radius_days=radius, snapshot_hash=self.audit["snapshot_hash"],
                            private_answer=private_answer(history, template_id, params, gold),
                            same_ticker_distractors=len(distractors),
                            same_broker_distractors=sum(record["firm"] in params["firms"] for record in distractors),
                            source_history_count=len(history)))
                        counts[size] += 1
                    audit.update(accepted=True, reason="accepted", base_id=base_id, parameters=copy.deepcopy(params), accepted_variants=len(survivors))
                finally:
                    attempt_log.append(audit)
            LOG.info("L%d final counts: %s", level, dict(sorted(counts.items())))
            if any(counts[size] < quota for size in config["sizes"]):
                LOG.warning("L%d shortfall; export will be blocked", level)
        return examples

    def verify_examples(self, examples) -> int:
        counts, registry, families, example_ids = Counter(), {}, defaultdict(list), set()
        for row in examples:
            require(row["example_id"] not in example_ids, "Duplicate example ID")
            example_ids.add(row["example_id"])
            size, level = row["context_size"], row["difficulty"]
            require(level in range(1, 6) and size in self.config["sizes"], "Unexpected stratum")
            counts[(level, size)] += 1
            records = row["evidence_list"]
            ids = [record["evidence_id"] for record in records]
            gold, distractors = set(row["golden_evidence_ids"]), set(row["distractor_ids"])
            require(len(ids) == size == len(set(ids)), "Context size/ID mismatch")
            require(bool(gold) and bool(distractors) and gold.isdisjoint(distractors) and gold | distractors == set(ids), "Invalid gold/distractor partition")
            require(len(gold) == row["gold_count"] == len(row["golden_evidence_ids"]), "Incorrect gold count")
            require(len(distractors) == len(row["distractor_ids"]), "Duplicate distractor IDs")
            params, template = row["parameters"], row["template_id"]
            require(row["cutoff"] == params["cutoff"], "Cutoff mismatch")
            require(row["difficulty"] == TEMPLATES[template]["level"], "Difficulty mismatch")
            require(row["question"] == question_text(template, params), "Question mismatch")
            require(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["cutoff"])), "Invalid cutoff")
            date.fromisoformat(row["cutoff"])
            for record in records:
                require(set(record) == set(EVIDENCE_FIELDS), "Evidence fields mismatch")
                require(all(usable(record[field]) for field in ["currentPriceTarget", "priorPriceTarget"]), "Invalid target")
                require(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", record["date"])) and record["date"] <= row["cutoff"], "Invalid/future evidence date")
                date.fromisoformat(record["date"])
                contents = {key: value for key, value in record.items() if key != "evidence_id"}
                require(checked_short_id(stable_hash(contents), registry, self.config["evidence_id_hex_chars"]) == record["evidence_id"], "Evidence hash mismatch")
            contract = golden_conditions(template, params)
            require(row["golden_conditions"] == contract, "Contract mismatch")
            require(row["golden_conditions_sql_like"] == golden_conditions_sql(contract), "Contract rendering mismatch")
            history = self.source_history(params)
            require(resolve_gold(history, template, params) == evaluate_golden_conditions(history, contract) == gold, "Source gold mismatch")
            require(resolve_gold(records, template, params) == evaluate_golden_conditions(records, contract) == gold, "Context gold mismatch")
            require(not deterministic_filter(records, template, params, row["cutoff"], gold, self.config), "Context filter failed")
            require(set(ids) == {record["evidence_id"] for record in natural_neighborhood(history, gold, row["adjacency_radius_days"])}, "Natural neighborhood changed")
            require(row["private_answer"] == private_answer(history, template, params, gold), "Private answer mismatch")
            require(row["snapshot_hash"] == self.audit["snapshot_hash"], "Snapshot mismatch")
            families[row["base_id"]].append({key: row[key] for key in ["context_size", "adjacency_radius_days", "split", "cutoff", "question"]} | {"ids": set(ids), "gold": gold})
        for group in families.values():
            require(len({row["context_size"] for row in group}) == len(group), "Repeated family size")
            for field in ["split", "cutoff", "question"]:
                require(len({row[field] for row in group}) == 1, "Inconsistent family " + field)
            previous, previous_gold = set(), None
            for row in sorted(group, key=lambda row: row["adjacency_radius_days"]):
                require(previous <= row["ids"] and (previous_gold is None or previous_gold == row["gold"]), "Family contexts/gold mismatch")
                previous, previous_gold = row["ids"], row["gold"]
        require(all(counts[(level, size)] == self.config["rows_per_stratum"] for level in range(1, 6) for size in self.config["sizes"]),
                "Dataset not balanced; existing export preserved. Counts: " + str(dict(sorted(counts.items()))))
        return len(example_ids)


def stream_items(path: Path, prefix: str):
    import ijson
    with path.open("rb") as stream:
        yield from ijson.items(stream, prefix, use_float=True)


def verify_export(path: Path, engine: Benchmark) -> int:
    manifest = next(stream_items(path, "manifest"))
    require(next(stream_items(path, "format_version")) == "benchmark-dataset-v5", "Unexpected export version")
    require(manifest["config"] == engine.config and manifest["source_audit"] == engine.audit, "Manifest mismatch")
    hashes = ArrayHash()
    def examples():
        for example in stream_items(path, "examples.item"):
            hashes.append(example)
            yield example
    count = engine.verify_examples(examples())
    require(hashes.hexdigest() == manifest["examples_sha256"], "Export example checksum mismatch")
    attempts = ArrayHash()
    for attempt in stream_items(path, "generation_attempts.item"):
        attempts.append(attempt)
    require(attempts.count == manifest["attempt_count"] and attempts.hexdigest() == manifest["attempts_sha256"], "Export audit checksum mismatch")
    return count


def save_export(engine: Benchmark, examples: list[dict], attempt_log: AttemptLog, destination: Path, provenance: dict) -> int:
    engine.verify_examples(examples)
    manifest = {
        "config": engine.config, "source_audit": engine.audit, "source_policy": SOURCE_POLICY,
        "download_provenance": provenance, "contains_private_gold": True,
        "examples_sha256": stable_hash(examples), "attempts_sha256": attempt_log.hash.hexdigest(),
        "attempt_count": attempt_log.hash.count,
        "proposal_policy": "Uniform cutoff day; random ticker/date; temporal questions sample firms with required date coverage. Templates alternate per difficulty. No context-size-dependent question proposals.",
        "stopping_policy": "Exactly rows_per_stratum per difficulty/context size. Full groups skipped; partial families allowed. Incomplete balance blocks export.",
        "attempt_log_policy": "All candidate and radius outcomes retained; full parameters retained for accepted candidates; rejected candidates reproducible from seed and deterministic proposal sequence.",
        "gold_contract": "golden_conditions is executable structured metadata; golden_conditions_sql_like is documentation, not production SQL.",
        "limitations": ["Latest/earliest refer to prefiltered complete-target reports.",
                       "Revision dates are not verified availability dates; aliases unmerged and targets not split-adjusted.",
                       "Exact-size quotas and temporal-coverage proposals introduce selection bias.",
                       "Template proportions within each stratum are not forced to be equal.",
                       "Quota selection retains partial families; size comparisons must use observed matched bases.",
                       "Gold-centered adjacency may reveal proximity cues; method neutrality is not guaranteed.",
                       "Different base questions may share evidence across splits.",
                       "Current constituent universe is not historical S&P 500 membership; Yahoo coverage is not guaranteed complete."],
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent, prefix="." + destination.name, delete=False) as output:
            temporary = Path(output.name)
            output.write(b'{"format_version":"benchmark-dataset-v5","manifest":')
            output.write(canonical(manifest))
            output.write(b',"examples":[')
            for index, example in enumerate(examples):
                if index:
                    output.write(b",")
                output.write(canonical(example))
            output.write(b'],"generation_attempts":[')
            attempt_log.stream.flush()
            attempt_log.stream.seek(0)
            for index, line in enumerate(attempt_log.stream):
                if index:
                    output.write(b",")
                output.write(line.rstrip(b"\n"))
            output.write(b"]}")
            output.flush()
            os.fsync(output.fileno())
        count = verify_export(temporary, engine)
        os.replace(temporary, destination)
        return count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fixture_record(evidence_id, day, firm="A", current=120.0, prior=100.0, ticker="TEST"):
    return dict(evidence_id=evidence_id, date=f"2020-01-{day:02d}", ticker=ticker, firm=firm, currentPriceTarget=current, priorPriceTarget=prior)


def run_self_tests() -> None:
    tests = 0
    params = dict(tickers=["TEST"], firms=["A"], cutoff="2020-01-31", date="2020-01-01", start="2020-01-01", end="2020-01-31", boundary="2020-01-15", threshold=10)
    rows = [fixture_record("a1", 1, current=100), fixture_record("a2", 15, current=120), fixture_record("a3", 31, current=140), fixture_record("b1", 1, firm="B", current=100), fixture_record("b2", 15, firm="B", current=110), fixture_record("b3", 31, firm="B", current=120), fixture_record("o1", 31, ticker="OTHER", current=200)]
    def check(template, data, settings, expected=None, error=None):
        nonlocal tests
        for resolver in [resolve_gold, lambda data, template, settings: evaluate_golden_conditions(data, golden_conditions(template, settings))]:
            try:
                actual = resolver(data, template, settings)
            except ValueError as caught:
                require(str(caught) == error, f"{template}: {caught} != {error}")
            else:
                require(error is None and actual == set(expected), f"Wrong answer: {template}")
            tests += 1
    check("L1_current", rows, params, ["a1"])
    check("L1_prior", rows, params, ["a1"])
    check("L2_raise_dollars", rows, params, ["a2", "a3", "b2", "b3"])
    check("L2_cut_percent", [fixture_record("cut", 1, current=90), fixture_record("small", 2, current=91), fixture_record("missing", 3, prior=None)], params, ["cut"])
    check("L3_two_firms", rows, dict(params, firms=["A", "B"]), ["a3", "b3"])
    check("L3_two_tickers", rows, dict(params, tickers=["TEST", "OTHER"]), ["a3", "o1"])
    check("L4_boundary", rows, params, ["a1", "a2"])
    check("L4_window", rows, params, ["a1", "a3"])
    check("L5_boundary", rows, dict(params, firms=["A", "B"]), ["a1", "a2", "b1", "b2"])
    check("L5_window", rows, dict(params, firms=["A", "B"]), ["a1", "a3", "b1", "b3"])
    check("L1_current", rows + [fixture_record("tie", 1)], params, error="ambiguous_same_day")
    check("L1_prior", [fixture_record("missing", 1, prior=None)], params, error="unavailable_selected_target")
    check("L4_window", [rows[0]], params, error="endpoints_not_distinct")
    check("L4_boundary", [rows[1]], params, error="missing_endpoint")
    check("L3_two_firms", rows + [fixture_record("missing_last", 31, current=None)], dict(params, firms=["A", "B"]), error="ambiguous_same_day")
    check("L3_two_firms", [fixture_record("valid_earlier", 1), fixture_record("invalid_latest", 31, current=None), rows[5]], dict(params, firms=["A", "B"]), error="unavailable_selected_target")
    check("L3_two_firms", rows, dict(params, cutoff="2020-01-15", firms=["A", "B"]), ["a2", "b2"])
    tied = [fixture_record("a1", 1, current=100), fixture_record("a2", 15, current=120), fixture_record("b1", 1, firm="B", current=200), fixture_record("b2", 15, firm="B", current=220)]
    check("L5_boundary", tied, dict(params, firms=["A", "B"]), error="tied_comparison")
    check("L2_raise_dollars", [fixture_record("none", 1, current=None)], params, [])
    require(private_answer(rows, "L5_boundary", dict(params, firms=["A", "B"]), {"a1", "a2", "b1", "b2"})["larger_signed_change_firm"] == "A", "Wrong winner")
    neighbors = [fixture_record("gold", 8), fixture_record("left", 1), fixture_record("right", 15), fixture_record("outside", 16)]
    require({row["evidence_id"] for row in natural_neighborhood(neighbors, {"gold"}, 7)} == {"gold", "left", "right"}, "Radius boundaries changed")
    projected = pd.DataFrame([{"GradeDate": f"2020-01-01T{hour}:00:00", "ticker": "TEST", "Firm": "A", "currentPriceTarget": 120, "priorPriceTarget": 100, "Action": action} for hour, action in [("01", "up"), ("23", "down")]])
    clean, audit = prepare_evidence(projected)
    require(len(clean) == 1 and audit["projected_duplicate_rows"] == 1, "Projection/dedup failed")
    require(set(clean[0]) == set(EVIDENCE_FIELDS), "Projection fields changed")
    rng, randomized = random.Random(9182), 0
    allowed_errors = {"missing_endpoint", "ambiguous_same_day", "endpoints_not_distinct", "tied_comparison", "unavailable_selected_target"}
    for case in range(100):
        records = [fixture_record(f"case{case}_{index}", rng.randint(1, 28), firm=rng.choice(["A", "B", "C"]), current=rng.choice([80., 100., 120.]), prior=rng.choice([90., 100.]), ticker=rng.choice(["TEST", "OTHER"])) for index in range(rng.randint(10, 50))]
        for template, spec in TEMPLATES.items():
            settings = dict(params, tickers=["TEST", "OTHER"] if template == "L3_two_tickers" else ["TEST"], firms=["A", "B"] if template == "L3_two_firms" or spec["level"] == 5 else ["A"], cutoff="2020-01-28", end="2020-01-28")
            contract = json.loads(json.dumps(golden_conditions(template, settings)))
            outcomes = []
            for resolver in [lambda: resolve_gold(records, template, settings), lambda: evaluate_golden_conditions(records, contract)]:
                try:
                    outcomes.append(("gold", resolver()))
                except ValueError as error:
                    require(str(error) in allowed_errors, "Unexpected contract error")
                    outcomes.append(("error", str(error)))
            require(outcomes[0][0] == outcomes[1][0], "Resolvers disagree on validity")
            if outcomes[0][0] == "gold":
                require(outcomes[0][1] == outcomes[1][1], "Randomized gold mismatch")
            randomized += 1
    source = pd.DataFrame([{"GradeDate": "2020-01-01", "ticker": "TEST", "Firm": "A", "currentPriceTarget": current, "priorPriceTarget": prior} for current, prior in [(120, 100), (None, 100), (120, None), (None, None), (0, 100), (120, float("inf"))]])
    clean, audit = prepare_evidence(source)
    require(len(clean) == 1 and audit["excluded_target_rows"] == 5, "Missing-target filter changed")
    contract = golden_conditions("L2_raise_dollars", params)
    records = [fixture_record("small", 1, current=110), fixture_record("large", 2, current=120)]
    require(evaluate_golden_conditions(records, contract) == {"small", "large"}, "Threshold boundary failed")
    contract["branches"][0]["value_filters"][0]["value"] = 20
    require(evaluate_golden_conditions(records, contract) == {"large"}, "Contract value mutation ignored")
    contract["branches"][0]["time_filters"][1]["value"] = "2020-01-01"
    require(evaluate_golden_conditions(records, contract) == set(), "Contract date mutation ignored")
    registry = {}
    checked_short_id("a" * 64, registry)
    try:
        checked_short_id("a" * 12 + "b" * 52, registry)
    except ValueError:
        pass
    else:
        raise ValueError("ID collision not detected")
    array_hash = ArrayHash()
    for item in [{"float": 1.2}, {"unicode": "firm", "null": None}]:
        array_hash.append(item)
    require(array_hash.hexdigest() == stable_hash([{"float": 1.2}, {"unicode": "firm", "null": None}]), "Streaming hash mismatch")
    LOG.info("Passed %d known-answer/error checks and %d randomized contracts; projection, adjacency, nulls, mutation, collision and hash checks passed", tests, randomized)


def load_revisions(path: Path, allow_partial: bool = False) -> tuple[pd.DataFrame, dict]:
    # Input revisions are much smaller than the full generation audit.
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    require(isinstance(document, dict) and document.get("format_version") == "broker-evidence-v1" and document.get("kind") == "revisions", "Expected broker_revisions.json from download_broker_evidence.py")
    metadata = document["metadata"]
    require(metadata.get("status") in {"complete", "partial"}, "Missing/invalid download status")
    require(allow_partial or metadata["status"] == "complete", "Partial download: resume downloader first or explicitly use --allow-partial")
    records = document["records"]
    require(isinstance(records, list) and len(records) == metadata["record_count"], "Revision record count mismatch")
    require(bool(records), "Revisions input is empty")
    return pd.DataFrame(records), metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revisions", type=Path, default=Path("data/broker_revisions.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_dataset_broker.json"))
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--rows-per-stratum", type=int, default=CONFIG["rows_per_stratum"])
    parser.add_argument("--max-candidates-per-level", type=int, default=CONFIG["max_candidates_per_level"])
    parser.add_argument("--sizes", type=int, nargs="+", default=CONFIG["sizes"])
    parser.add_argument("--progress-every", type=int, default=CONFIG["progress_every"])
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly allow partial download metadata")
    parser.add_argument("--self-test", action="store_true", help="Run offline rule tests only; writes no dataset")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if min(args.rows_per_stratum, args.max_candidates_per_level, args.progress_every) < 1:
        parser.error("quotas, candidate budget and progress interval must be positive")
    if not args.sizes or any(size < 5 for size in args.sizes) or len(set(args.sizes)) != len(args.sizes):
        parser.error("sizes must be distinct integers >= 5 (L5 requires four gold records plus a distractor)")
    run_self_tests()
    if args.self_test:
        return 0
    require(args.revisions.resolve() != args.output.resolve(), "Output must not overwrite revisions input")
    source, provenance = load_revisions(args.revisions, args.allow_partial)
    config = copy.deepcopy(CONFIG)
    config.update(seed=args.seed, rows_per_stratum=args.rows_per_stratum,
                  max_candidates_per_level=args.max_candidates_per_level, sizes=sorted(args.sizes), progress_every=args.progress_every)
    evidence, audit = prepare_evidence(source, config["evidence_id_hex_chars"])
    del source
    LOG.info("Source audit: %s", json.dumps(audit, sort_keys=True))
    engine = Benchmark(evidence, audit, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Temporary audit stays on the destination filesystem, not memory-backed /tmp.
    with tempfile.TemporaryFile(mode="w+b", dir=args.output.parent) as stream:
        attempt_log = AttemptLog(stream)
        examples = engine.generate(attempt_log)
        LOG.info("Candidate summary: %s", dict(attempt_log.reasons))
        count = save_export(engine, examples, attempt_log, args.output, provenance)
    LOG.info("Exported and reloaded %d validated examples to %s (%.1fs)", count, args.output, time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted; existing benchmark preserved. Re-run with the same seed to reproduce generation.")
        raise SystemExit(130)
    except Exception as error:
        LOG.error("Benchmark failed; existing output preserved: %s", error)
        raise SystemExit(1)
