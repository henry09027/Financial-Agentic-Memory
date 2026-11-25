#!/usr/bin/env python3
"""Step 3b: MCP-style semantic-layer benchmark (question -> SQLite SQL over the supplied evidence).

    python run_mcp_benchmark.py --input artifacts/benchmarks/broker.json --smoke-test
    python run_mcp_benchmark.py --input artifacts/benchmarks/edgar.json
    python run_mcp_benchmark.py --input artifacts/benchmarks/edgar.json --resume
    python run_mcp_benchmark.py --self-test          # CPU, fake model, both domains

Conditions (paired on identical examples):
  baseline        schema-derived prompt only
  semantic_model  the same prompt plus the domain's portable semantic-model YAML
                  (default: semantic_models/<domain>.yaml beside this script)
The model sees the question and the table schema, never evidence rows, gold labels,
golden SQL or private answers. SQL runs locally on an isolated, read-only in-memory
SQLite table holding exactly that example's evidence (authorizer whitelist, time and
row limits). Invalid SQL is recorded, not repaired. No MCP server is deployed.
The table DDL and column notes come from the benchmark's evidence_schema, so EDGAR and
broker benchmarks run through the same code.

Shared with run_knowledge_graph_benchmark.py (retrieval_common.py): CLI, balanced
selection, vLLM backend, journal + fingerprinted --resume, statuses, scoring, results.json /
results.csv / summary.json / summary.csv under artifacts/results/<benchmark>/mcp/<split>/.
Exit codes: 0 complete, 1 error, 130 interrupted (resume).
Requirements: pip install -r requirements-gpu.txt (PyYAML for every run).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sqlite3
import re
import tempfile
import time

from pipeline_common import PIPELINE_DIR, evidence_fields, file_sha256, get_logger, require, run_main, setup_logging
from retrieval_common import (Method, RunContext, add_benchmark_arguments, final_answer_text, input_error_result,
                              make_result, result_key, run_benchmark, run_stage, stage_failure, validate_benchmark_args,
                              validate_example)

LOG = get_logger("mcp")
CONDITIONS = ("baseline", "semantic_model")
SQL_TYPES = {"string": "TEXT", "date": "TEXT", "number": "REAL", "integer": "INTEGER"}
BASE_PROMPT = """You translate evidence-retrieval questions into SQLite SQL.
Return exactly ONE read-only SQL query, with one result column named evidence_id.
The question may request a JSON list: ignore only that output-format request and return SQL instead, with no explanation or markdown.
Use only the evidence table with this schema:
{ddl}
Columns:
{columns}
Data notes:
{notes}
Dates are ISO YYYY-MM-DD strings. Use SQLite syntax, not QUALIFY.
Return all and only the requested supporting rows, each evidence_id once. For comparisons return the supporting rows specified by the question, not an aggregate answer or only a winner.
Use the date bounds and selection rules stated in the question. The table contains exactly that question's supplied evidence.
Do not invent evidence IDs. Do not change data, inspect system tables, attach files, or load extensions."""
SEMANTIC_SUFFIX = "\n\nAdditional evidence semantic model (YAML):\n"
ALLOWED_FUNCTIONS = {"min", "max", "abs", "round", "count", "sum", "avg", "coalesce", "nullif", "row_number", "rank",
                     "dense_rank", "first_value", "last_value", "lag", "lead", "date", "julianday", "strftime",
                     "unixepoch", "like", "glob", "ifnull", "iif", "lower", "upper", "length", "substr", "substring",
                     "trim", "ltrim", "rtrim", "typeof", "nth_value", "ntile"}


def table_ddl(schema: dict) -> str:
    columns = [f'"{field["name"]}" {SQL_TYPES[field["type"]]}' + (" PRIMARY KEY" if field["name"] == schema["id_field"] else "")
               for field in schema["fields"]]
    return "CREATE TABLE evidence (" + ", ".join(columns) + ");"


def system_prompt(schema: dict, semantic_text: str | None) -> str:
    prompt = BASE_PROMPT.format(ddl=table_ddl(schema),
                                columns="\n".join(f"- {field['name']}: {field['description']}" for field in schema["fields"]),
                                notes="\n".join(f"- {note}" for note in schema["notes"]))
    return prompt + (SEMANTIC_SUFFIX + semantic_text if semantic_text is not None else "")


def extract_sql(raw: str, thinking: bool) -> tuple[str, bool]:
    text, fenced = final_answer_text(raw, thinking)
    if not re.match(r"^(SELECT|WITH)\b", text, re.I):
        raise ValueError("Expected a single SELECT or WITH query")
    return text, fenced


def execute_sql(sql: str, evidence: list[dict], schema: dict, args) -> tuple[list[str], float, float]:
    started = time.perf_counter()
    fields = evidence_fields(schema)
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(table_ddl(schema))
        quoted = ", ".join(f'"{name}"' for name in fields)
        connection.executemany(f"INSERT INTO evidence ({quoted}) VALUES ({', '.join('?' for _ in fields)})",
                               ([row[name] for name in fields] for row in evidence))
        connection.commit()
        connection.execute("PRAGMA query_only=ON")
        connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 50000)
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1000000)
        connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100)

        def authorize(action, arg1, arg2, database, trigger):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and arg1 == "evidence":
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or arg1 or "").lower() in ALLOWED_FUNCTIONS:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorize)
        deadline = time.perf_counter() + args.sql_timeout
        connection.set_progress_handler(lambda: int(time.perf_counter() > deadline), 1000)
        setup_seconds = time.perf_counter() - started
        query_started = time.perf_counter()
        cursor = connection.execute(sql)  # execute(), not executescript(): one statement only
        if not cursor.description or [column[0] for column in cursor.description] != ["evidence_id"]:
            raise ValueError("SQL must return exactly one column named evidence_id")
        rows = cursor.fetchmany(args.max_result_rows + 1)
        if len(rows) > args.max_result_rows:
            raise ValueError("SQL result limit exceeded")
        ids = [row[0] for row in rows]
        if any(not isinstance(value, str) for value in ids):
            raise ValueError("SQL returned a non-string evidence_id")
        return ids, setup_seconds, time.perf_counter() - query_started
    finally:
        connection.close()


def load_semantic_model(path: Path, domain: str) -> str:
    import yaml
    require(path.is_file(), f"Semantic model not found: {path}. Put the domain's YAML at semantic_models/{domain}.yaml "
                            "or pass --semantic-model")
    text = path.read_text(encoding="utf-8")
    semantic = yaml.safe_load(text)
    require(isinstance(semantic, dict) and semantic.get("format_version") == "portable-evidence-semantics-v1",
            f"{path}: expected a portable-evidence-semantics-v1 YAML specification")
    require(semantic.get("physical_table") == "evidence" and semantic.get("dialect") == "sqlite",
            f"{path}: semantic model must describe the SQLite evidence table")
    require(semantic.get("domain", domain) == domain, f"{path}: semantic model domain {semantic.get('domain')!r} != benchmark {domain!r}")
    return text


def setup(context: RunContext) -> None:
    args, domain = context.args, context.benchmark.domain
    path = (args.semantic_model or args.semantic_model_dir / f"{domain}.yaml").resolve()
    context.state["semantic_text"] = load_semantic_model(path, domain)
    context.state["prompts"] = {CONDITIONS[0]: system_prompt(context.schema, None),
                                CONDITIONS[1]: system_prompt(context.schema, context.state["semantic_text"])}
    context.fingerprint_extra.update(semantic_model_path=str(path), semantic_model_sha256=file_sha256(path))


def process(context: RunContext, batch: list[tuple[dict, str]]) -> list[dict]:
    schema, args = context.schema, context.args
    results, valid = {}, []
    for example, condition in batch:
        try:
            validate_example(example, schema)
            valid.append((example, condition))
        except Exception as error:  # noqa: BLE001 - recorded as input_error
            results[result_key(example, condition)] = input_error_result(context, example, condition, error)
    stages = run_stage(context, [(result_key(example, condition),
                                  [{"role": "system", "content": context.state["prompts"][condition]},
                                   {"role": "user", "content": example["question"]}])
                                 for example, condition in valid], "retriever", args.max_tokens)
    for example, condition in valid:
        key = result_key(example, condition)
        stage, started = stages[key], time.perf_counter()
        artifacts, predicted, status, error = {"sql": None}, [], "ok", None
        if stage["status"] != "ok":
            status, error = stage_failure("retriever", stage)
        else:
            phase = "retriever"
            try:
                artifacts["sql"], artifacts["fenced_output"] = extract_sql(stage["raw_output"], args.thinking)
                phase = "sql_execution"
                predicted, artifacts["database_setup_seconds"], artifacts["sql_seconds"] = execute_sql(
                    artifacts["sql"], example["evidence_list"], schema, args)
            except Exception as error_:  # noqa: BLE001
                status = "invalid_output" if phase == "retriever" else "execution_error"
                error = {"stage": phase, "type": type(error_).__name__, "message": str(error_)[:500]}
        results[key] = make_result(context, example, condition, status=status, error=error, predicted=predicted,
                                   stages={"retriever": stage}, artifacts=artifacts,
                                   postprocessing_seconds=time.perf_counter() - started)
    return [results[result_key(example, condition)] for example, condition in batch]


def fake_responder(messages: list[dict], call: int) -> str:
    if call % 5 == 0:
        return "SELECT nope FROM evidence"
    sql = "SELECT evidence_id FROM evidence ORDER BY evidence_id LIMIT 2"
    return f"```sql\n{sql}\n```" if call % 3 == 0 else sql


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--semantic-model", type=Path, help="YAML file; default: <--semantic-model-dir>/<domain>.yaml")
    parser.add_argument("--semantic-model-dir", type=Path, default=PIPELINE_DIR / "semantic_models")
    parser.add_argument("--sql-timeout", type=float, default=2.0, help="Seconds per query")
    parser.add_argument("--max-result-rows", type=int, default=1000)


METHOD = Method(name="mcp", script=Path(__file__), conditions=CONDITIONS,
                prompts={"base_prompt_template": BASE_PROMPT, "semantic_suffix": SEMANTIC_SUFFIX,
                         "allowed_functions": sorted(ALLOWED_FUNCTIONS)},
                process=process, responder=fake_responder, setup=setup, add_arguments=add_arguments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_benchmark_arguments(parser)
    METHOD.add_arguments(parser)
    args = parser.parse_args()
    validate_benchmark_args(parser, args)
    if not math.isfinite(args.sql_timeout) or args.sql_timeout <= 0 or args.max_result_rows < 1:
        parser.error("--sql-timeout and --max-result-rows must be positive")
    setup_logging(args.log_level)
    if args.self_test:
        with tempfile.TemporaryDirectory() as directory:
            for domain in ("broker", "edgar"):
                Path(directory, f"{domain}.yaml").write_text(
                    f"format_version: portable-evidence-semantics-v1\ndomain: {domain}\nphysical_table: evidence\n"
                    "dialect: sqlite\n", encoding="utf-8")
            args.semantic_model, args.semantic_model_dir = None, Path(directory)
            return run_benchmark(METHOD, args, LOG)
    return run_benchmark(METHOD, args, LOG)


if __name__ == "__main__":
    run_main(main, LOG)
