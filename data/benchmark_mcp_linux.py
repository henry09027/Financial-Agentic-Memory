#!/usr/bin/env python3
"""Paired SQLite retrieval benchmark for Linux; no Snowflake dependencies.

Python 3.11+ and a CUDA GPU compatible with the installed vLLM build are required.
Install in a separate virtual environment (select a torch/CUDA build for your driver):
    python3 -m venv .venv
    . .venv/bin/activate
    pip install 'vllm==0.34.0' PyYAML ijson

Smoke test: ten questions total, two per difficulty when five levels are present,
with BOTH modes evaluated (twenty responses):
    python benchmark_mcp_linux.py --input benchmark_dataset.json --smoke-test 10

Full test split (default); use --split all for every dataset example:
    python benchmark_mcp_linux.py --input benchmark_dataset.json --batch-size 16
    python benchmark_mcp_linux.py --input benchmark_dataset.json --split all

Two GPUs, with explicitly selected Linux device ordinals:
    CUDA_VISIBLE_DEVICES=0,1 python benchmark_mcp_linux.py --input benchmark_dataset.json \
        --tensor-parallel-size 2 --batch-size 32 --max-num-seqs 32

Resume: repeat the SAME invocation and add --resume. Output defaults to
benchmark_dataset_mcp.json in the current directory. Existing outputs are protected.
Use a new --output path to start a different experiment, NOT --resume.
The adjacent *.checkpoint.jsonl journal preserves metadata, batch timings, every
returned raw response, and scored results. Do not delete it until finished.

vLLM batches multiple prompts through one engine. It does NOT create a model per
question. GPU tensor parallelism is optional. SQL runs locally on isolated SQLite
connections. Invalid SQL is recorded, not repaired. Fatal engine failures stop the
run after writing a partial output; resume reruns only requests without a durable
raw response. A batch whose engine failed may have consumed unobservable tokens;
these are not invented. Concurrent runs targeting the same output are locked out.

Neither mode sees evidence rows, gold labels, golden SQL, or private answers.
The semantic mode adds ONLY the portable YAML to the unchanged baseline prompt.
This YAML is informed by previous pilot errors, so reusing those questions is a
DEVELOPMENT comparison, not a fresh holdout evaluation. No MCP server is deployed.

Accuracy is exact evidence-ID SET equality. Duplicate IDs are recorded separately.
Timing under concurrency differs from the old sequential notebook: report batch
wall time and throughput, and per-request engine latency only if vLLM exposes it.
Never interpret batch_wall_seconds / batch_size as measured request latency.
This script and its YAML were authored without running inference in Snowflake.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import sys
import time
import traceback
import uuid

MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODES = ("without_semantic_model", "with_semantic_model")
COLUMNS = ("evidence_id", "date", "ticker", "firm", "currentPriceTarget", "priorPriceTarget")
SCHEMA = "CREATE TABLE evidence (evidence_id TEXT PRIMARY KEY, date TEXT, ticker TEXT, firm TEXT, currentPriceTarget REAL, priorPriceTarget REAL);"
BASE_PROMPT = """You translate evidence-retrieval questions into SQLite SQL.
Return exactly ONE read-only SQL query, with one result column named evidence_id.
The question may request a JSON list: ignore only that output-format request and return SQL instead, with no explanation or markdown.
Use only the evidence table with this schema:
CREATE TABLE evidence (evidence_id TEXT PRIMARY KEY, date TEXT, ticker TEXT, firm TEXT, currentPriceTarget REAL, priorPriceTarget REAL);
Dates are ISO YYYY-MM-DD strings. All targets are available positive numbers.
Match ticker and firm exactly, case sensitively. Use SQLite syntax, not QUALIFY.
Return all and only requested supporting reports, each evidence_id once. For comparisons return the supporting reports specified by the question, not an aggregate answer or only a winner.
Use the date bounds and selection rules stated in the question. The table contains exactly that question's supplied evidence.
Do not invent evidence IDs. Do not change data, inspect system tables, attach files, or load extensions."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--semantic-model", type=Path, default=Path(__file__).with_name("evidence_semantic_model.yaml"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_dataset_mcp.json"))
    parser.add_argument("--smoke-test", type=int, metavar="N", help="Balanced N-question sample; 10 means 20 paired responses")
    parser.add_argument("--split", choices=("test", "dev", "train", "all"), default="test")
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Hugging Face revision; default matches the original pilot")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--sql-timeout", type=float, default=2.0)
    parser.add_argument("--max-result-rows", type=int, default=1000)
    parser.add_argument("--thinking", action="store_true", help="Enable thinking identically in both modes")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs, as in the original pilot")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("batch_size", "max_num_seqs", "tensor_parallel_size", "max_model_len", "max_tokens", "max_result_rows"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.smoke_test is not None and args.smoke_test < 1:
        parser.error("--smoke-test must be positive")
    if not math.isfinite(args.sql_timeout) or args.sql_timeout <= 0:
        parser.error("--sql-timeout must be finite and positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if args.max_tokens >= args.max_model_len:
        parser.error("--max-tokens must be less than --max-model-len")
    if args.batch_size > args.max_num_seqs:
        print("Note: batch-size exceeds max-num-seqs; engine latency includes scheduler queueing.", file=sys.stderr)
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.semantic_model = args.semantic_model.resolve()
    if args.output in (args.input, args.semantic_model, Path(__file__).resolve()):
        parser.error("Output must not overwrite input, semantic model, or script")
    if not args.output.parent.is_dir():
        parser.error("Output directory must already exist")
    return args


def examples(path: Path, split: str):
    import ijson
    with path.open("rb") as source:
        # Streaming handles the ~300 MB dataset without materializing its audit log.
        for index, example in enumerate(ijson.items(source, "examples.item", use_float=True)):
            if not isinstance(example, dict):
                raise ValueError(f"Dataset examples[{index}] is not an object")
            if split == "all" or example.get("split") == split:
                yield index, example


def hash_rank(seed: int, *values) -> str:
    return hashlib.sha256(json.dumps([seed, *values], ensure_ascii=False).encode()).hexdigest()


def select_examples(args):
    counts = collections.Counter()
    seen_ids = set()
    candidates = collections.defaultdict(dict)
    for index, row in examples(args.input, args.split):
        identifier = row.get("example_id")
        level = row.get("difficulty")
        if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
            raise ValueError(f"Missing or duplicate example_id at source index {index}")
        if not isinstance(level, int) or isinstance(level, bool):
            raise ValueError(f"Invalid difficulty at source index {index}")
        seen_ids.add(identifier)
        counts[level] += 1
        if args.smoke_test:
            base = row.get("base_id", identifier)
            if not isinstance(base, str) or not base:
                raise ValueError(f"Invalid base_id for {identifier}")
            rank = hash_rank(args.seed, level, base)
            variant = hash_rank(args.seed, identifier)
            bucket = candidates[level]
            current = bucket.get(base)
            if current is None or variant < current[1]:
                bucket[base] = (rank, variant, index, identifier, base)
            # Keep only N best bases per difficulty, at most N per level in memory.
            if len(bucket) > args.smoke_test:
                del bucket[max(bucket, key=lambda key: bucket[key][0])]
    if not counts:
        raise ValueError("No examples found for the requested split")
    if not args.smoke_test:
        return None, sum(counts.values()), dict(counts)
    queues = {level: sorted(bucket.values()) for level, bucket in candidates.items()}
    chosen = []
    used_bases = set()
    while len(chosen) < args.smoke_test:
        changed = False
        for level in sorted(queues):
            while queues[level] and queues[level][0][4] in used_bases:
                queues[level].pop(0)
            if queues[level] and len(chosen) < args.smoke_test:
                candidate = queues[level].pop(0)
                chosen.append(candidate)
                used_bases.add(candidate[4])
                changed = True
        if not changed:
            raise ValueError("Not enough distinct base questions for --smoke-test")
    return {item[2] for item in chosen}, len(chosen), dict(counts)


def batches(iterator, size):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def validate_example(row):
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        raise ValueError("Question must be a nonempty string")
    evidence = row.get("evidence_list")
    gold = row.get("golden_evidence_ids")
    if not isinstance(evidence, list) or not isinstance(gold, list):
        raise ValueError("evidence_list and golden_evidence_ids must be arrays")
    ids = set()
    for item in evidence:
        if not isinstance(item, dict) or any(column not in item for column in COLUMNS):
            raise ValueError("Evidence row missing schema columns")
        for column in COLUMNS[:4]:
            if not isinstance(item[column], str) or not item[column]:
                raise ValueError(f"Invalid evidence {column}")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", item["date"]):
            raise ValueError("Evidence date must be ISO YYYY-MM-DD")
        dt.date.fromisoformat(item["date"])
        for column in COLUMNS[4:]:
            value = item[column]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid evidence {column}")
        if item["evidence_id"] in ids:
            raise ValueError("Duplicate evidence_id")
        ids.add(item["evidence_id"])
    if any(not isinstance(value, str) for value in gold):
        raise ValueError("Invalid golden evidence ID")
    if len(gold) != len(set(gold)) or not set(gold) <= ids:
        raise ValueError("Gold IDs must be unique members of the evidence table")


def extract_sql(text):
    text = text.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    elif "<think>" in text:
        raise ValueError("Incomplete reasoning; no final SQL")
    match = re.fullmatch(r"```(?:sqlite|sql)?[ \t]*\n?([\s\S]*?)\s*```", text, re.I)
    if match:
        text = match.group(1).strip()
    if not re.match(r"^(SELECT|WITH)\b", text, re.I):
        raise ValueError("Expected a single SELECT or WITH query")
    return text


def execute_sql(sql, evidence, args):
    started = time.perf_counter()
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(SCHEMA)
        connection.executemany("INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?)",
                               ([row[column] for column in COLUMNS] for row in evidence))
        connection.commit()
        connection.execute("PRAGMA query_only=ON")
        connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 50000)
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1000000)
        connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100)
        allowed = {"min", "max", "abs", "round", "count", "sum", "avg", "coalesce", "nullif",
                   "row_number", "rank", "dense_rank", "first_value", "last_value", "lag", "lead",
                   "date", "julianday", "strftime", "unixepoch", "like", "glob", "ifnull", "iif",
                   "lower", "upper", "length", "substr", "substring", "trim", "ltrim", "rtrim",
                   "typeof", "nth_value", "ntile"}

        def authorize(action, arg1, arg2, database, trigger):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and arg1 == "evidence":
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or arg1 or "").lower() in allowed:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorize)
        deadline = time.perf_counter() + args.sql_timeout
        connection.set_progress_handler(lambda: int(time.perf_counter() > deadline), 1000)
        setup_seconds = time.perf_counter() - started
        query_started = time.perf_counter()
        # execute(), not executescript(), rejects multiple statements.
        cursor = connection.execute(sql)
        if not cursor.description or [column[0] for column in cursor.description] != ["evidence_id"]:
            raise ValueError("SQL must return exactly one column named evidence_id")
        rows = cursor.fetchmany(args.max_result_rows + 1)
        if len(rows) > args.max_result_rows:
            raise ValueError("SQL result limit exceeded")
        ids = [row[0] for row in rows]
        valid = {row["evidence_id"] for row in evidence}
        if any(not isinstance(value, str) or value not in valid for value in ids):
            raise ValueError("Unknown or invalid returned evidence_id")
        return list(dict.fromkeys(ids)), len(ids) - len(set(ids)), setup_seconds, time.perf_counter() - query_started
    finally:
        connection.close()


def metrics(predicted, gold, success):
    predicted, gold = set(predicted), set(gold)
    matched = len(predicted & gold)
    precision = matched / len(predicted) if predicted else float(not gold)
    recall = matched / len(gold) if gold else float(not predicted)
    return {"exact_match": bool(success and predicted == gold),
            "precision": precision if success else 0.0, "recall": recall if success else 0.0,
            "f1": 2 * precision * recall / (precision + recall) if success and precision + recall else 0.0}


def key_for(index, mode):
    return f"{index}:{mode}"


class Journal:
    """Append-only durable events; only a torn final line is discarded on resume."""
    def __init__(self, path, resume):
        self.path = path
        self.header = None
        self.done = set()
        self.raw_pending = {}
        if resume:
            if not path.is_file():
                raise FileNotFoundError("--resume requires the matching checkpoint journal")
            with path.open("r+b") as source:
                while True:
                    offset = source.tell()
                    line = source.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        source.truncate(offset)
                        break
                    event = json.loads(line)
                    if event["type"] == "header":
                        if self.header is not None:
                            raise ValueError("Duplicate checkpoint header")
                        self.header = event["data"]
                    elif event["type"] == "raw":
                        if event["data"]["key"] in self.done:
                            raise ValueError("Raw response follows completed result in checkpoint")
                        self.raw_pending[event["data"]["key"]] = event["data"]
                    elif event["type"] == "result":
                        key = event["data"]["key"]
                        if key in self.done:
                            raise ValueError("Duplicate completed checkpoint record")
                        self.done.add(key)
                        self.raw_pending.pop(key, None)
            if self.header is None:
                raise ValueError("Checkpoint has no complete header")
        self.sink = path.open("a" if resume else "x", encoding="utf-8")

    def append(self, kind, data):
        event = {"type": kind, "at_utc": utc_now(), "data": data}
        self.sink.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.sink.flush()
        os.fsync(self.sink.fileno())
        if kind == "header":
            self.header = data
        elif kind == "raw":
            self.raw_pending[data["key"]] = data
        elif kind == "result":
            self.done.add(data["key"])
            self.raw_pending.pop(data["key"], None)

    def close(self):
        self.sink.close()


def score(row, raw, args):
    started = time.perf_counter()
    result = {field: row.get(field) for field in ("example_id", "base_id", "difficulty", "context_size", "template_id", "split", "question", "evidence_list", "golden_evidence_ids")}
    result.update(raw)
    result.update(sql=None, retrieved_evidence_ids=[], retrieved_evidence=[], duplicate_ids_removed=0,
                  sql_execution_success=False, database_setup_seconds=None, sql_seconds=None)
    stage = "input_validation"
    try:
        validate_example(row)
        if raw.get("error"):
            result["error"] = raw["error"]
        else:
            stage = "sql_extraction"
            if raw.get("finish_reason") == "length":
                raise ValueError("Output truncated at max_tokens; not executing partial SQL")
            result["sql"] = extract_sql(raw["raw_output"])
            stage = "sql_execution"
            ids, duplicates, setup, execution = execute_sql(result["sql"], row["evidence_list"], args)
            evidence_map = {item["evidence_id"]: item for item in row["evidence_list"]}
            result.update(retrieved_evidence_ids=ids, retrieved_evidence=[evidence_map[value] for value in ids],
                          duplicate_ids_removed=duplicates, sql_execution_success=True,
                          database_setup_seconds=setup, sql_seconds=execution, error=None)
    except Exception as error:
        result["error"] = {"stage": stage, "type": type(error).__name__, "message": str(error)}
    result["evaluation_eligible"] = not (result.get("error") and result["error"]["stage"] == "input_validation")
    result["metrics"] = (metrics(result["retrieved_evidence_ids"], row["golden_evidence_ids"], result["sql_execution_success"])
                         if result["evaluation_eligible"] else None)
    result["postprocessing_seconds"] = time.perf_counter() - started
    latency = raw.get("engine_latency_seconds")
    result["retrieval_phase_seconds"] = latency + result["postprocessing_seconds"] if latency is not None else None
    return result


def summarize(records):
    eligible = [row for row in records if row["metrics"] is not None]
    latencies = [row["engine_latency_seconds"] for row in records if row.get("engine_latency_seconds") is not None]
    return {"records": len(records), "scored_records": len(eligible),
            "invalid_input_records": len(records) - len(eligible),
            "exact_matches": sum(row["metrics"]["exact_match"] for row in eligible),
            "accuracy": statistics.mean(row["metrics"]["exact_match"] for row in eligible) if eligible else None,
            "sql_success_rate": statistics.mean(row["sql_execution_success"] for row in eligible) if eligible else None,
            **{name + "_macro": statistics.mean(row["metrics"][name] for row in eligible) if eligible else None
               for name in ("precision", "recall", "f1")},
            "input_tokens_observed": sum(row.get("input_tokens") or 0 for row in records),
            "output_tokens_observed": sum(row.get("output_tokens") or 0 for row in records),
            "token_counts_complete": all(row.get("input_tokens") is not None and row.get("output_tokens") is not None for row in records),
            "engine_latency_available_count": len(latencies),
            "engine_latency_seconds_mean": statistics.mean(latencies) if latencies else None,
            "engine_latency_seconds_p95": sorted(latencies)[math.ceil(.95 * len(latencies)) - 1] if latencies else None,
            "errors_by_stage": dict(collections.Counter(row["error"]["stage"] for row in records if row.get("error")))}


def assemble_output(journal, output, status, fatal_error=None):
    # Keep only compact metrics in memory. Large evidence and prompt fields stream to disk.
    compact = []
    batch_events = []
    sessions = []
    response_tokens = collections.defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "responses": 0})
    failures = []
    with journal.path.open(encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)
            data = event["data"]
            if event["type"] == "result":
                compact.append({name: data.get(name) for name in ("key", "source_index", "mode", "difficulty", "metrics", "sql_execution_success", "error", "input_tokens", "output_tokens", "engine_latency_seconds")})
            elif event["type"] == "batch":
                batch_events.append(data)
            elif event["type"] == "runtime":
                sessions.append(data)
            elif event["type"] == "fatal":
                failures.append(data)
            elif event["type"] == "raw" and data.get("raw_output") is not None:
                totals = response_tokens[data["mode"]]
                totals["responses"] += 1
                totals["input_tokens"] += data.get("input_tokens") or 0
                totals["output_tokens"] += data.get("output_tokens") or 0
    summary = {}
    for mode in MODES:
        rows = [row for row in compact if row["mode"] == mode]
        mode_batches = [batch for batch in batch_events if batch["mode"] == mode]
        seconds = sum(batch["wall_seconds"] for batch in mode_batches)
        generated = sum(batch["returned_responses"] for batch in mode_batches)
        summary[mode] = {**summarize(rows), "generation_batch_wall_seconds": seconds,
                         "generated_requests_per_second": generated / seconds if seconds else None,
                         "observed_output_tokens_per_batch_second": response_tokens[mode]["output_tokens"] / seconds if seconds else None,
                         "all_observed_response_tokens": dict(response_tokens[mode]),
                         "by_difficulty": {str(level): summarize([row for row in rows if row["difficulty"] == level])
                                           for level in sorted({row["difficulty"] for row in rows})}}
    paired = collections.defaultdict(dict)
    for row in compact:
        if row["metrics"] is not None:
            paired[row["source_index"]][row["mode"]] = row["metrics"]["exact_match"]
    pairs = [pair for pair in paired.values() if all(mode in pair for mode in MODES)]
    paired_summary = {"scored_pairs": len(pairs),
                      "both_correct": sum(all(pair.values()) for pair in pairs),
                      "semantic_only_correct": sum(not pair[MODES[0]] and pair[MODES[1]] for pair in pairs),
                      "baseline_only_correct": sum(pair[MODES[0]] and not pair[MODES[1]] for pair in pairs),
                      "both_wrong": sum(not any(pair.values()) for pair in pairs)}
    header = {"format_version": "paired-sql-retrieval-v2", "status": status, "updated_at_utc": utc_now(),
              "run": journal.header, "runtime_sessions": sessions, "summary": summary,
              "paired_comparison": paired_summary, "completed_responses": len(compact),
              "expected_responses": journal.header["selected_question_count"] * 2,
              "raw_responses_pending_scoring": len(journal.raw_pending), "fatal_error": fatal_error,
              "failure_history": failures, "generation_batches": batch_events,
              "limitations": ["Semantic rules were informed by previous pilot errors; inspected cases are development data.",
                              "Evidence rows and golden labels are never in generation prompts.",
                              "Semantic YAML is prompt context, not a deployed Snowflake view.",
                              "Concurrent request latency is not comparable to isolated sequential latency.",
                              "retrieval_phase_seconds sums engine latency and postprocessing, excluding host batch-return waiting.",
                              "Missing engine metrics remain null; batch time is never divided into fabricated per-request latency.",
                              "Fatal engine failures can consume unobservable tokens; only observed token counts are recorded.",
                              "Partial-run accuracy covers scored records only; check expected and completed counts.",
                              "Warmup tokens and initialization are recorded separately from benchmark token totals."]}
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as sink:
        sink.write(json.dumps(header, ensure_ascii=False, indent=2, allow_nan=False)[:-1])
        sink.write(',\n"results": [\n')
        first = True
        with journal.path.open(encoding="utf-8") as source:
            for line in source:
                event = json.loads(line)
                if event["type"] == "result":
                    if not first:
                        sink.write(",\n")
                    sink.write(json.dumps(event["data"], ensure_ascii=False, allow_nan=False))
                    first = False
        sink.write("\n]}\n")
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, output)
    print(json.dumps({"status": status, "completed_responses": len(compact), "summary": summary}), flush=True)


class Generator:
    def __init__(self, args, journal):
        import torch
        from vllm import LLM, SamplingParams
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.tensor_parallel_size:
            raise RuntimeError("Insufficient CUDA devices for requested tensor parallelism")
        self.args = args
        self.params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)
        started = time.perf_counter()
        self.llm = LLM(model=MODEL, revision=journal.header["resolved_revision"],
                       tokenizer_revision=journal.header["resolved_revision"], dtype="bfloat16",
                       tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
                       max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_memory_utilization,
                       enforce_eager=args.enforce_eager, enable_prefix_caching=False,
                       seed=args.seed, limit_mm_per_prompt={"image": 0, "video": 0})
        self.tokenizer = self.llm.get_tokenizer()
        runtime = {"initialization_seconds": time.perf_counter() - started,
                   "gpu_names": [torch.cuda.get_device_name(index) for index in range(args.tensor_parallel_size)],
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "versions": {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers", "huggingface-hub", "PyYAML", "ijson")},
                   "sqlite_version": sqlite3.sqlite_version, "python": sys.version,
                   "request_latency_source": "RequestOutput.metrics.finished_time - arrival_time; null when unavailable",
                   "request_latency_includes_scheduler_queueing": True}
        warmup = self.render([{"role": "user", "content": "Return only this SQL: SELECT evidence_id FROM evidence LIMIT 1;"}])
        started = time.perf_counter()
        response = self.llm.generate([warmup], self.params, use_tqdm=False)[0]
        runtime.update(warmup_seconds=time.perf_counter() - started,
                       warmup_input_tokens=len(response.prompt_token_ids),
                       warmup_output_tokens=len(response.outputs[0].token_ids))
        journal.append("runtime", runtime)

    def render(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=self.args.thinking)


def request_latency(response):
    timing = getattr(response, "metrics", None)
    arrival = getattr(timing, "arrival_time", None)
    finished = getattr(timing, "finished_time", None)
    if all(isinstance(value, (int, float)) and math.isfinite(value) for value in (arrival, finished)) and finished >= arrival:
        return finished - arrival
    return None


def process_mode(batch, mode, semantic_text, args, journal, generator):
    pending = []
    for index, row in batch:
        key = key_for(index, mode)
        if key in journal.done:
            continue
        if key in journal.raw_pending:
            journal.append("result", score(row, journal.raw_pending[key], args))
            continue
        raw = {"key": key, "source_index": index, "example_id": row["example_id"], "mode": mode,
               "raw_output": None, "rendered_prompt": None, "input_tokens": None, "output_tokens": None,
               "engine_latency_seconds": None, "finish_reason": None, "error": None}
        try:
            validate_example(row)
        except Exception as error:
            raw["error"] = {"stage": "input_validation", "type": type(error).__name__, "message": str(error)}
            journal.append("raw", raw)
            journal.append("result", score(row, raw, args))
            continue
        if generator is None:
            generator = Generator(args, journal)
        system = BASE_PROMPT
        if mode == MODES[1]:
            system += "\n\nAdditional evidence semantic model (YAML):\n" + semantic_text
        raw["messages"] = [{"role": "system", "content": system}, {"role": "user", "content": row["question"]}]
        try:
            started = time.perf_counter()
            raw["rendered_prompt"] = generator.render(raw["messages"])
            length = len(generator.tokenizer.encode(raw["rendered_prompt"], add_special_tokens=False))
            raw["prompt_render_seconds"] = time.perf_counter() - started
            if length + args.max_tokens > args.max_model_len:
                raise ValueError(f"Prompt ({length}) plus output budget exceeds context length; no truncation performed")
        except Exception as error:
            raw["error"] = {"stage": "prompt_preparation", "type": type(error).__name__, "message": str(error)}
            journal.append("raw", raw)
            journal.append("result", score(row, raw, args))
            continue
        pending.append((row, raw))
    if not pending:
        return generator
    batch_id = uuid.uuid4().hex
    journal.append("batch_start", {"batch_id": batch_id, "mode": mode, "keys": [raw["key"] for _, raw in pending]})
    started = time.perf_counter()
    try:
        outputs = generator.llm.generate([raw["rendered_prompt"] for _, raw in pending], generator.params, use_tqdm=False)
    except Exception as error:
        journal.append("batch", {"batch_id": batch_id, "mode": mode, "wall_seconds": time.perf_counter() - started,
                                 "requested_responses": len(pending), "returned_responses": 0,
                                 "error": str(error), "token_usage_unknown": True})
        # An engine may be dead after an OOM. Do not pretend it is safe to keep using it.
        raise RuntimeError("vLLM batch failed; partial output saved. Resume after resolving the engine error.") from error
    wall = time.perf_counter() - started
    journal.append("batch", {"batch_id": batch_id, "mode": mode, "wall_seconds": wall,
                             "requested_responses": len(pending), "returned_responses": len(outputs), "error": None})
    if len(outputs) != len(pending):
        journal.append("unexpected_responses", {"batch_id": batch_id, "responses": [repr(item) for item in outputs]})
        raise RuntimeError("vLLM returned an unexpected response count; cannot safely pair responses")
    # vLLM LLM.generate returns outputs in input order. Persist ALL raw outputs before executing SQL.
    for (row, raw), response in zip(pending, outputs):
        raw.update(batch_id=batch_id, batch_wall_seconds=wall, batch_size=len(pending),
                   engine_latency_seconds=request_latency(response), vllm_request_id=str(response.request_id))
        if not response.outputs:
            raw["error"] = {"stage": "generation", "type": "EmptyCompletion", "message": "No completion returned"}
        else:
            completion = response.outputs[0]
            raw.update(raw_output=completion.text,
                       input_tokens=len(response.prompt_token_ids) if response.prompt_token_ids is not None else None,
                       output_tokens=len(completion.token_ids), finish_reason=completion.finish_reason,
                       stop_reason=completion.stop_reason)
        journal.append("raw", raw)
    for row, raw in pending:
        journal.append("result", score(row, raw, args))
    print(f"Completed {len(journal.done)}/{journal.header['selected_question_count'] * 2} responses", flush=True)
    return generator


def run(args):
    import yaml
    semantic_text = args.semantic_model.read_text(encoding="utf-8")
    semantic = yaml.safe_load(semantic_text)
    if not isinstance(semantic, dict) or semantic.get("format_version") != "portable-evidence-semantics-v1":
        raise ValueError("Expected a portable-evidence-semantics-v1 YAML specification")
    if semantic.get("physical_table") != "evidence" or semantic.get("dialect") != "sqlite":
        raise ValueError("Semantic model must describe SQLite evidence")
    config = {key: value for key, value in vars(args).items() if key not in {"resume", "input", "output", "semantic_model"}}
    fingerprints = {"input_sha256": digest_file(args.input), "yaml_sha256": digest_file(args.semantic_model),
                    "script_sha256": digest_file(Path(__file__)), "configuration": config}
    fingerprint = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
    checkpoint = args.output.with_name(args.output.name + ".checkpoint.jsonl")
    if not args.resume and (args.output.exists() or checkpoint.exists()):
        raise FileExistsError("Output/checkpoint exists. Use --resume or a different --output path.")
    selected, total, counts = select_examples(args)
    revision = None
    if not args.resume:
        # Resolve before creating a journal so network failure cannot strand an empty checkpoint.
        from huggingface_hub import HfApi
        revision = HfApi().model_info(MODEL, revision=args.revision).sha
    with contextlib.closing(Journal(checkpoint, args.resume)) as journal:
        if args.resume:
            if journal.header["fingerprint"] != fingerprint:
                raise ValueError("Resume rejected: input, YAML, script, or configuration changed")
        else:
            journal.append("header", {"created_at_utc": utc_now(), "model": MODEL, "resolved_revision": revision,
                                      "fingerprint": fingerprint, **fingerprints, "semantic_model_yaml": semantic_text,
                                      "baseline_prompt": BASE_PROMPT, "input_path": str(args.input),
                                      "selected_source_indices": sorted(selected) if selected is not None else None,
                                      "selected_question_count": total, "available_split_counts": counts,
                                      "selection_policy": "Seeded hash-ranked unique bases, balanced round-robin by difficulty for smoke; otherwise all examples in split",
                                      "mode_policy": "Both modes on identical questions; alternate which mode runs first each batch",
                                      "prefix_caching": False, "temperature": 0.0, "automatic_sql_repair": False})
        status, fatal, code = "complete", None, 0
        generator = None
        try:
            source = ((index, row) for index, row in examples(args.input, args.split) if selected is None or index in selected)
            for number, batch in enumerate(batches(source, args.batch_size)):
                modes = MODES if number % 2 == 0 else tuple(reversed(MODES))
                for mode in modes:
                    generator = process_mode(batch, mode, semantic_text, args, journal, generator)
            if len(journal.done) != total * 2:
                raise RuntimeError("Completed response count does not match the selected paired workload")
        except KeyboardInterrupt:
            status, code = "interrupted", 130
            fatal = {"type": "KeyboardInterrupt", "message": "Resume to continue incomplete requests"}
            journal.append("fatal", fatal)
        except Exception as error:
            status, code = "failed", 1
            fatal = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            journal.append("fatal", fatal)
            print(fatal["traceback"], file=sys.stderr)
        finally:
            assemble_output(journal, args.output, status, fatal)
        return code


def main():
    args = parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    lock_path = args.output.with_name(args.output.name + ".lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another process is using this output path", file=sys.stderr)
            return 1
        try:
            return run(args)
        except Exception as error:
            print(f"Preflight/export failure: {type(error).__name__}: {error}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
