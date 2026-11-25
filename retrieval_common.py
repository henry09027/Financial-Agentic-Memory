#!/usr/bin/env python3
"""Shared harness for the retrieval benchmarks (run_knowledge_graph_benchmark.py, run_mcp_benchmark.py).

One definition of: CLI flags, example selection, the vLLM backend and CPU fake backend,
the append-only journal with fingerprinted resume, output parsing, scoring, the result
record, summaries, and the run-directory layout:

    artifacts/results/<benchmark>/<method>/<split>[_perlevelN][_fake]/
        run.log            log of every session
        journal.jsonl      append-only events: header, runtime, batch, stage (raw model I/O), result, fatal
        run_metadata.json  configuration, fingerprint, environment, status
        results.json       fam-envelope-v1 kind=retrieval_results, one record per (example, condition)
        results.csv        flat view of results.json
        summary.json/.csv  accuracy, P/R/F1, tokens, latency by condition x {overall, difficulty,
                           context size, template}, plus a paired comparison of the two conditions

Scoring (identical for every method): predictions are deduplicated in order;
exact_match = predicted set equals gold set; strict_match additionally requires no
duplicate and no unknown IDs; unknown IDs are counted as extra (never a crash); a
failed pipeline scores 0. Input-validation failures are excluded from denominators
and reported separately. Engine latency comes only from vLLM request metrics; batch
wall time is reported per batch and NEVER divided into per-request latency.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import sys
import tempfile
import time
from typing import Any, Callable
import uuid

from pipeline_common import (ARTIFACTS_DIR, BENCHMARK_SCHEMA, DEFAULT_SEED, EVIDENCE_SCHEMAS, EXIT_ERROR,
                             EXIT_INTERRUPTED, EXIT_OK, EnvelopeReader, PipelineError,
                             add_common_arguments, atomic_write_json, exclusive_lock, file_sha256, read_envelope,
                             require, setup_logging, stable_hash, utc_now, validate_evidence_row,
                             validate_evidence_schema, write_envelope)

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
STATUSES = ("ok", "input_error", "context_overflow", "truncated", "generation_error", "invalid_output", "execution_error")
LIMITATIONS = [
    "Accuracy is exact evidence-ID set equality; see strict_match for duplicate/unknown-ID-free answers.",
    "Partial runs cover completed results only; compare completed_results with expected_results.",
    "Concurrent batching makes request latency include scheduler queueing; it is not isolated sequential latency.",
    "Missing engine metrics stay null; token counts are only those observed.",
    "Retrieval methods see only pipeline_common.retriever_payload(); gold and private answers never enter prompts.",
]


class EngineFailure(RuntimeError):
    """The inference engine failed for every request in a batch; stop and resume later."""


class Terminated(Exception):
    pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=False,
                        help="Benchmark file, e.g. artifacts/benchmarks/broker.json or edgar.json (required unless --self-test)")
    parser.add_argument("--run-dir", type=Path, help="Default: artifacts/results/<benchmark>/<method>/<split>[_perlevelN]")
    parser.add_argument("--resume", action="store_true", help="Continue the run in --run-dir; configuration must be identical")
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="test")
    parser.add_argument("--per-level", type=int, help="Balanced sample: N distinct base questions per difficulty")
    parser.add_argument("--smoke-test", action="store_true", help="Shortcut for --per-level 2 (10 questions for 5 levels)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--backend", choices=["vllm", "fake"], default="vllm",
                        help="'fake' is a CPU harness check with meaningless scores")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Hugging Face revision; a full commit SHA is reproducible")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=32, help="Concurrent sequences in the engine")
    parser.add_argument("--batch-size", type=int, default=32, help="(example, condition) tasks per batch")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Output budget of the final retrieval stage")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", action="store_true", help="Enable the model's thinking mode identically in all conditions")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs")
    parser.add_argument("--prefix-caching", action="store_true", help="Enable vLLM prefix caching (off by default for comparable timing)")
    add_common_arguments(parser, overwrite=False)


def validate_benchmark_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.self_test and args.input is None:
        parser.error("--input is required")
    if args.smoke_test:
        if args.per_level is not None and args.per_level != 2:
            parser.error("--smoke-test conflicts with --per-level")
        args.per_level = 2
    for name in ("tensor_parallel_size", "max_model_len", "max_num_seqs", "batch_size", "max_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.per_level is not None and args.per_level < 1:
        parser.error("--per-level must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and nonnegative")
    if args.resume and args.run_dir is None and args.input is None:
        parser.error("--resume needs the same --input (and --run-dir if one was given)")


# ---------------------------------------------------------------------------
# Benchmark loading and selection
# ---------------------------------------------------------------------------
@dataclass
class Benchmark:
    path: Path
    file_sha256: str
    metadata: dict
    schema: dict
    domain: str
    examples: list[dict]
    selection: dict


def load_benchmark(path: Path, split: str, per_level: int | None, seed: int, logger: logging.Logger) -> Benchmark:
    """Two streaming passes: select by lightweight fields, then load only selected examples."""
    light, seen = [], set()
    with EnvelopeReader(path, "benchmark", BENCHMARK_SCHEMA) as reader:
        metadata = reader.metadata
        schema = validate_evidence_schema(metadata.get("evidence_schema"))
        domain = metadata.get("domain")
        require(domain in EVIDENCE_SCHEMAS, f"Unknown benchmark domain {domain!r}")
        for index, example in enumerate(reader):
            require(isinstance(example, dict), f"examples[{index}] is not an object")
            identifier = example.get("example_id")
            require(isinstance(identifier, str) and identifier not in seen, f"Missing or duplicate example_id at {index}")
            seen.add(identifier)
            light.append((index, identifier, example.get("base_id"), example.get("difficulty"),
                          example.get("template_id"), example.get("context_size"), example.get("split")))
    if not metadata.get("complete", False):
        logger.warning("Benchmark metadata says complete=false (generated with --allow-shortfall)")
    available = {}
    for row in light:
        available[row[6]] = available.get(row[6], 0) + 1
    pool = [row for row in light if split == "all" or row[6] == split]
    require(pool, f"No examples in split {split!r}")
    if per_level is None:
        chosen = {row[0] for row in pool}
        policy = f"all examples in split {split}"
    else:
        chosen = set()
        for level in sorted({row[3] for row in pool}):
            families: dict[str, dict[str, list]] = {}
            for row in pool:
                if row[3] == level:
                    families.setdefault(row[4], {}).setdefault(row[2], []).append(row)
            queues = {template: sorted(bases, key=lambda base: stable_hash([seed, "base", base]))
                      for template, bases in families.items()}
            used_sizes: dict[int, int] = {}
            picked = 0
            while picked < per_level:
                progressed = False
                for template in sorted(queues):
                    if picked == per_level or not queues[template]:
                        continue
                    variants = families[template][queues[template].pop(0)]
                    fewest = min(used_sizes.get(row[5], 0) for row in variants)
                    variant = min((row for row in variants if used_sizes.get(row[5], 0) == fewest),
                                  key=lambda row: stable_hash([seed, "variant", row[1]]))
                    chosen.add(variant[0])
                    used_sizes[variant[5]] = used_sizes.get(variant[5], 0) + 1
                    picked += 1
                    progressed = True
                require(progressed, f"Only {picked} distinct base questions at difficulty {level}; lower --per-level")
        policy = (f"{per_level} distinct base questions per difficulty from split {split}: templates in rotation, bases "
                  "hash-ranked by seed, one context variant per base balancing context sizes")
    examples = []
    with EnvelopeReader(path, "benchmark", BENCHMARK_SCHEMA) as reader:
        for index, example in enumerate(reader):
            if index in chosen:
                examples.append(example)
    selection = {"split": split, "per_level": per_level, "seed": seed, "policy": policy,
                 "selected_examples": len(examples), "available_by_split": available}
    return Benchmark(path, file_sha256(path), metadata, schema, domain, examples, selection)


def validate_example(example: dict, schema: dict) -> None:
    require(isinstance(example.get("question"), str) and example["question"].strip(), "Question must be a nonempty string")
    evidence, gold = example.get("evidence_list"), example.get("golden_evidence_ids")
    require(isinstance(evidence, list) and evidence and isinstance(gold, list), "evidence_list/golden_evidence_ids must be arrays")
    ids = set()
    for row in evidence:
        validate_evidence_row(row, schema)
        require(row[schema["id_field"]] not in ids, "Duplicate evidence_id")
        ids.add(row[schema["id_field"]])
    require(all(isinstance(value, str) for value in gold) and len(gold) == len(set(gold)) and set(gold) <= ids,
            "Gold IDs must be unique members of the evidence list")


# ---------------------------------------------------------------------------
# Output parsing and scoring
# ---------------------------------------------------------------------------
FENCE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n?([\s\S]*?)\s*```")


def final_answer_text(raw: str, thinking: bool) -> tuple[str, bool]:
    """Text after the last </think>; one wrapping Markdown fence is format noise and is removed (and recorded)."""
    text = raw.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    elif thinking or "<think>" in text:
        raise ValueError("No completed final answer after reasoning")
    match = FENCE.fullmatch(text)
    if match:
        return match.group(1).strip(), True
    return text, False


def strict_json(text: str) -> Any:
    def reject_constant(value):
        raise ValueError("Non-JSON numeric constant: " + value)

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def parse_id_list(text: str) -> list[str]:
    value = strict_json(text)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Expected a JSON list of evidence_id strings")
    return value


def score_predictions(predicted: list[str], gold: list[str], supplied: list[str], success: bool) -> dict:
    predicted = predicted if success else []
    actual, gold_set, supplied_set = set(predicted), set(gold), set(supplied)
    hits = len(actual & gold_set)
    precision = hits / len(actual) if actual else float(not gold_set)
    recall = hits / len(gold_set) if gold_set else 1.0
    duplicates, invalid = len(predicted) - len(actual), sorted(actual - supplied_set)
    exact = bool(success and actual == gold_set)
    return {"exact_match": exact, "strict_match": exact and duplicates == 0 and not invalid,
            "precision": precision if success else 0.0, "recall": recall if success else 0.0,
            "f1": 2 * precision * recall / (precision + recall) if success and precision + recall else 0.0,
            "predicted_count": len(actual), "gold_count": len(gold_set), "missed_ids": sorted(gold_set - actual),
            "extra_ids": sorted(actual - gold_set), "invalid_ids": invalid, "duplicate_count": duplicates}


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def new_stage() -> dict:
    return {"status": None, "invoked": False, "input_tokens": None, "output_tokens": None, "finish_reason": None,
            "engine_latency_seconds": None, "raw_output": None, "error": None}


def request_latency(response) -> float | None:
    timing = getattr(response, "metrics", None)
    arrival, finished = getattr(timing, "arrival_time", None), getattr(timing, "finished_time", None)
    if all(isinstance(value, (int, float)) and math.isfinite(value) for value in (arrival, finished)) and finished >= arrival:
        return finished - arrival
    return None


class VLLMBackend:
    """One in-process engine; whole batches are submitted and continuously batched on the GPU(s)."""

    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        try:
            import torch
            from vllm import LLM, SamplingParams
        except ImportError as error:
            raise PipelineError(f"vLLM/torch not importable ({error}); pip install -r requirements-gpu.txt") from error
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.tensor_parallel_size:
            raise PipelineError("Insufficient CUDA devices for the requested tensor parallelism")
        self.args, self.logger = args, logger
        started = time.perf_counter()
        if Path(args.model).is_dir():
            revision, self.resolved_revision = None, "local-directory-not-content-hashed"
        else:
            try:
                from huggingface_hub import HfApi
                revision = self.resolved_revision = HfApi().model_info(args.model, revision=args.revision, timeout=30).sha
            except Exception as error:  # noqa: BLE001
                if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
                    raise PipelineError(f"Cannot resolve revision {args.revision!r} ({error}); pass a full commit SHA") from error
                logger.warning("Hugging Face lookup failed (%s); using pinned revision %s from cache", error, args.revision)
                revision = self.resolved_revision = args.revision
        kwargs = dict(model=args.model, revision=revision, tokenizer_revision=revision, trust_remote_code=False,
                      dtype=args.dtype, seed=args.seed, tensor_parallel_size=args.tensor_parallel_size,
                      gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
                      max_num_seqs=args.max_num_seqs, enforce_eager=args.enforce_eager,
                      enable_prefix_caching=args.prefix_caching, generation_config="vllm",
                      limit_mm_per_prompt={"image": 0, "video": 0})
        for _ in range(3):
            try:
                self.engine = LLM(**kwargs)
                break
            except TypeError as error:
                optional = [key for key in ("limit_mm_per_prompt", "generation_config") if key in str(error) and key in kwargs]
                if not optional:
                    raise
                logger.warning("This vLLM version rejects %s; retrying without it", optional[0])
                kwargs.pop(optional[0])
        self.SamplingParams = SamplingParams
        self.tokenizer = self.engine.get_tokenizer()
        self.info = {
            "backend": "vllm", "model": args.model, "requested_revision": args.revision,
            "resolved_revision": self.resolved_revision, "engine_kwargs": {key: value for key, value in kwargs.items()},
            "load_seconds": time.perf_counter() - started,
            "hardware": [{"name": torch.cuda.get_device_name(index),
                          "total_vram_gib": round(torch.cuda.get_device_properties(index).total_memory / 2**30, 2)}
                         for index in range(torch.cuda.device_count())],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "versions": {name: _version(name) for name in ("vllm", "torch", "transformers", "huggingface-hub", "PyYAML")},
            "request_latency_source": "RequestOutput.metrics.finished_time - arrival_time; null when unavailable",
        }
        warmup = self._prompt_ids([{"role": "user", "content": "Reply with []"}])
        started = time.perf_counter()
        response = self._run([warmup], 8)[0]
        self.info.update(warmup_seconds=time.perf_counter() - started, warmup_input_tokens=len(response.prompt_token_ids),
                         warmup_output_tokens=len(response.outputs[0].token_ids) if response.outputs else None)

    def _prompt_ids(self, messages: list[dict]) -> list[int]:
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=self.args.thinking)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return list(ids)

    def _run(self, prompts: list[list[int]], max_tokens: int):
        params = self.SamplingParams(temperature=self.args.temperature, max_tokens=max_tokens, seed=self.args.seed)
        return self.engine.generate([{"prompt_token_ids": ids} for ids in prompts], params, use_tqdm=False)

    @staticmethod
    def _fill(stage: dict, response) -> None:
        stage.update(invoked=True, engine_latency_seconds=request_latency(response),
                     input_tokens=len(response.prompt_token_ids) if response.prompt_token_ids is not None else None)
        if not response.outputs:
            stage.update(status="generation_error", error={"type": "EmptyCompletion", "message": "No completion returned"})
            return
        completion = response.outputs[0]
        stage.update(output_tokens=len(completion.token_ids), raw_output=completion.text,
                     finish_reason=completion.finish_reason,
                     status="truncated" if completion.finish_reason == "length" else "ok")

    @staticmethod
    def _clear_cuda() -> None:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def generate(self, messages_list: list[list[dict]], max_tokens: int) -> tuple[list[dict], dict]:
        stages, runnable = [new_stage() for _ in messages_list], []
        for index, messages in enumerate(messages_list):
            try:
                ids = self._prompt_ids(messages)
            except Exception as error:  # noqa: BLE001
                stages[index].update(status="generation_error", error={"type": type(error).__name__, "message": f"Tokenization failed: {error}"})
                continue
            stages[index]["prepared_input_tokens"] = len(ids)
            if len(ids) + max_tokens > self.args.max_model_len:
                stages[index].update(status="context_overflow", error={
                    "type": "ContextOverflow",
                    "message": f"{len(ids)} prompt + {max_tokens} output tokens > max_model_len {self.args.max_model_len}; no truncation"})
            else:
                runnable.append((index, ids))
        info = {"requested": len(messages_list), "submitted": len(runnable), "mode": "batched", "error": None}
        started = time.perf_counter()
        if runnable:
            try:
                responses = self._run([ids for _, ids in runnable], max_tokens)
                require(len(responses) == len(runnable), "vLLM returned an unexpected number of responses")
                for (index, _), response in zip(runnable, responses):
                    self._fill(stages[index], response)
            except Exception as error:  # noqa: BLE001 - isolate one bad request (or OOM) from the batch
                self.logger.warning("Batch of %d failed (%s: %s); retrying one at a time", len(runnable), type(error).__name__, error)
                info.update(mode="individual_retry", error=f"{type(error).__name__}: {error}")
                self._clear_cuda()
                failures = 0
                for index, ids in runnable:
                    try:
                        self._fill(stages[index], self._run([ids], max_tokens)[0])
                    except Exception as single:  # noqa: BLE001
                        failures += 1
                        stages[index].update(status="generation_error", invoked=True,
                                             error={"type": type(single).__name__, "message": str(single)})
                        self._clear_cuda()
                if failures == len(runnable):
                    raise EngineFailure("Every request failed individually; the engine is likely unusable") from error
        info.update(wall_seconds=time.perf_counter() - started, returned=sum(stage["invoked"] for stage in stages))
        return stages, info


class FakeBackend:
    """CPU stand-in: deterministic responder plus periodic truncation to exercise failure paths."""

    def __init__(self, responder: Callable[[list[dict], int], str]):
        self.responder, self.calls = responder, 0
        self.info = {"backend": "fake", "model": "fake", "resolved_revision": "fake", "versions": {}, "hardware": []}

    def generate(self, messages_list: list[list[dict]], max_tokens: int) -> tuple[list[dict], dict]:
        started, stages = time.perf_counter(), []
        for messages in messages_list:
            self.calls += 1
            text = self.responder(messages, self.calls)
            stage = new_stage()
            stage.update(status="ok", invoked=True, input_tokens=len(json.dumps(messages)) // 4, output_tokens=len(text) // 4,
                         raw_output=text, finish_reason="stop")
            if self.calls % 13 == 0:
                stage.update(status="truncated", finish_reason="length")
            stages.append(stage)
        return stages, {"requested": len(messages_list), "submitted": len(messages_list), "returned": len(messages_list),
                        "mode": "batched", "error": None, "wall_seconds": time.perf_counter() - started}


def _version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Journal (append-only, fsynced; only a torn final line is discarded on resume)
# ---------------------------------------------------------------------------
class Journal:
    def __init__(self, path: Path, resume: bool):
        self.path, self.header = path, None
        self.stages: dict[tuple[str, str], dict] = {}
        self.results: dict[str, dict] = {}
        if resume:
            require(path.is_file(), f"--resume requires an existing journal at {path}")
            with path.open("r+b") as source:
                while True:
                    offset = source.tell()
                    line = source.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        source.truncate(offset)
                        break
                    self._absorb(json.loads(line))
            require(self.header is not None, "Journal has no header")
        else:
            require(not path.exists(), f"{path} exists; pass --resume to continue it or choose another --run-dir")
        self._sink = path.open("a" if resume else "x", encoding="utf-8")

    def _absorb(self, event: dict) -> None:
        kind, data = event["type"], event["data"]
        if kind == "header":
            require(self.header is None, "Duplicate journal header")
            self.header = data
        elif kind == "stage":
            self.stages[(data["key"], data["stage"])] = data
        elif kind == "result":
            self.results[data["result_key"]] = data

    def append(self, kind: str, data: dict) -> None:
        event = {"type": kind, "at_utc": utc_now(), "data": data}
        self._sink.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self._sink.flush()
        os.fsync(self._sink.fileno())
        self._absorb(event)

    def events(self, kind: str):
        with self.path.open(encoding="utf-8") as source:
            for line in source:
                event = json.loads(line)
                if event["type"] == kind:
                    yield event["data"]

    def close(self) -> None:
        self._sink.close()


# ---------------------------------------------------------------------------
# Method plumbing
# ---------------------------------------------------------------------------
@dataclass
class Method:
    name: str
    script: Path
    conditions: tuple[str, str]
    prompts: dict
    process: Callable[["RunContext", list[tuple[dict, str]]], list[dict]]
    responder: Callable[[list[dict], int], str]
    setup: Callable[["RunContext"], None] = lambda context: None
    add_arguments: Callable[[argparse.ArgumentParser], None] = lambda parser: None


@dataclass
class RunContext:
    args: argparse.Namespace
    method: Method
    benchmark: Benchmark
    journal: Journal | None
    logger: logging.Logger
    state: dict = field(default_factory=dict)
    fingerprint_extra: dict = field(default_factory=dict)
    _backend: Any = None

    @property
    def schema(self) -> dict:
        return self.benchmark.schema

    def backend(self):
        if self._backend is None:
            self._backend = (FakeBackend(self.method.responder) if self.args.backend == "fake"
                             else VLLMBackend(self.args, self.logger))
            self.journal.append("runtime", dict(self._backend.info, python=sys.version, platform=platform.platform()))
        return self._backend


def result_key(example: dict, condition: str) -> str:
    return f"{example['example_id']}::{condition}"


def run_stage(context: RunContext, items: list[tuple[str, list[dict]]], stage: str, max_tokens: int) -> dict[str, dict]:
    """Generate one stage for many tasks, reusing journaled outputs (except generation errors) on resume."""
    output, pending = {}, []
    for key, messages in items:
        cached = context.journal.stages.get((key, stage))
        if cached is not None and cached["status"] != "generation_error":
            output[key] = dict(cached, cached=True)
        else:
            pending.append((key, messages))
    if pending:
        records, info = context.backend().generate([messages for _, messages in pending], max_tokens)
        batch_id = uuid.uuid4().hex
        context.journal.append("batch", dict(info, batch_id=batch_id, stage=stage))
        for (key, messages), record in zip(pending, records):
            record.update(key=key, stage=stage, batch_id=batch_id, messages=messages)
            context.journal.append("stage", record)
            output[key] = dict(record, cached=False)
    return output


def stage_failure(stage_name: str, stage: dict) -> tuple[str, dict]:
    error = stage.get("error") or {"type": stage["status"], "message": f"finish_reason={stage.get('finish_reason')}"}
    return stage["status"], dict(error, stage=stage_name)


def make_result(context: RunContext, example: dict, condition: str, *, status: str, error: dict | None,
                predicted: list[str], stages: dict[str, dict], artifacts: dict, postprocessing_seconds: float) -> dict:
    require(status in STATUSES, f"Unknown status {status}")
    ok = status == "ok"
    supplied = [row[context.schema["id_field"]] for row in example.get("evidence_list") or [] if isinstance(row, dict)]
    metrics = None if status == "input_error" else score_predictions(predicted, example.get("golden_evidence_ids") or [], supplied, ok)
    invoked = [stage for stage in stages.values() if stage.get("invoked")]

    def total(name):
        values = [stage.get(name) for stage in invoked]
        return None if any(value is None for value in values) else sum(values)
    summary_stages = {name: {key: stage.get(key) for key in ("status", "invoked", "cached", "input_tokens", "output_tokens",
                                                            "finish_reason", "engine_latency_seconds", "raw_output",
                                                            "error", "batch_id")}
                      for name, stage in stages.items()}
    return {"result_key": result_key(example, condition), "example_id": example.get("example_id"),
            "base_id": example.get("base_id"), "domain": context.benchmark.domain, "difficulty": example.get("difficulty"),
            "template_id": example.get("template_id"), "context_size": example.get("context_size"),
            "split": example.get("split"), "method": context.method.name, "condition": condition, "status": status,
            "error": error, "predicted_evidence_ids": list(dict.fromkeys(predicted)) if ok else [],
            "golden_evidence_ids": example.get("golden_evidence_ids"), "metrics": metrics, "stages": summary_stages,
            "artifacts": artifacts, "tokens": {"input": total("input_tokens"), "output": total("output_tokens")},
            "timing": {"engine_latency_seconds": total("engine_latency_seconds"),
                       "postprocessing_seconds": postprocessing_seconds}}


def input_error_result(context: RunContext, example: dict, condition: str, error: Exception) -> dict:
    return make_result(context, example, condition, status="input_error",
                       error={"stage": "input_validation", "type": type(error).__name__, "message": str(error)},
                       predicted=[], stages={}, artifacts={}, postprocessing_seconds=0.0)


# ---------------------------------------------------------------------------
# Summaries and exports
# ---------------------------------------------------------------------------
def aggregate(rows: list[dict]) -> dict:
    scored = [row for row in rows if row["metrics"] is not None]
    latencies = [row["timing"]["engine_latency_seconds"] for row in rows if row["timing"]["engine_latency_seconds"] is not None]

    def mean(name):
        return statistics.mean(row["metrics"][name] for row in scored) if scored else None
    return {"results": len(rows), "scored": len(scored), "input_errors": len(rows) - len(scored),
            "correct": sum(row["metrics"]["exact_match"] for row in scored), "accuracy": mean("exact_match"),
            "strict_accuracy": mean("strict_match"), "precision": mean("precision"), "recall": mean("recall"),
            "f1": mean("f1"), "ok_rate": statistics.mean(row["status"] == "ok" for row in rows) if rows else None,
            "status_counts": {status: sum(row["status"] == status for row in rows) for status in STATUSES},
            "input_tokens": sum(row["tokens"]["input"] or 0 for row in rows),
            "output_tokens": sum(row["tokens"]["output"] or 0 for row in rows),
            "token_counts_complete": all(row["tokens"]["input"] is not None and row["tokens"]["output"] is not None
                                         for row in rows if row["status"] != "input_error"),
            "engine_latency_available": len(latencies),
            "engine_latency_seconds_mean": statistics.mean(latencies) if latencies else None,
            "engine_latency_seconds_p95": sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1] if latencies else None}


def summarize(results: list[dict], conditions: tuple[str, str], batches: list[dict]) -> dict:
    groups = {"overall": lambda row: "all", "difficulty": lambda row: row["difficulty"],
              "context_size": lambda row: row["context_size"], "template_id": lambda row: row["template_id"]}
    table = []
    for group_by, key in groups.items():
        values = sorted({key(row) for row in results}, key=str)
        for value in values:
            for condition in conditions:
                rows = [row for row in results if row["condition"] == condition and key(row) == value]
                if rows:
                    table.append({"group_by": group_by, "group_value": value, "condition": condition, **aggregate(rows)})
    paired = {}
    for row in results:
        if row["metrics"] is not None:
            paired.setdefault(row["example_id"], {})[row["condition"]] = row["metrics"]["exact_match"]
    pairs = [pair for pair in paired.values() if all(condition in pair for condition in conditions)]
    first, second = conditions
    comparison = {"conditions": list(conditions), "scored_pairs": len(pairs),
                  "both_correct": sum(pair[first] and pair[second] for pair in pairs),
                  f"only_{first}_correct": sum(pair[first] and not pair[second] for pair in pairs),
                  f"only_{second}_correct": sum(pair[second] and not pair[first] for pair in pairs),
                  "both_wrong": sum(not pair[first] and not pair[second] for pair in pairs)}
    throughput = {}
    for batch in batches:
        entry = throughput.setdefault(batch["stage"], {"batches": 0, "wall_seconds": 0.0, "returned_responses": 0})
        entry["batches"] += 1
        entry["wall_seconds"] += batch.get("wall_seconds") or 0.0
        entry["returned_responses"] += batch.get("returned") or 0
    for entry in throughput.values():
        entry["responses_per_second"] = entry["returned_responses"] / entry["wall_seconds"] if entry["wall_seconds"] else None
    return {"table": table, "paired_comparison": comparison, "throughput_by_stage": throughput,
            "timing_note": "Batch wall time is per batch; it is never divided into per-request latency."}


CSV_COLUMNS = ["result_key", "example_id", "base_id", "domain", "difficulty", "template_id", "context_size", "split",
               "method", "condition", "status", "error_stage", "error_message", "exact_match", "strict_match", "precision",
               "recall", "f1", "predicted_count", "gold_count", "duplicate_count", "invalid_count", "input_tokens",
               "output_tokens", "engine_latency_seconds", "postprocessing_seconds"]


def flat_result(row: dict) -> dict:
    metrics = row["metrics"] or {}
    error = row["error"] or {}
    return {"result_key": row["result_key"], "example_id": row["example_id"], "base_id": row["base_id"],
            "domain": row["domain"], "difficulty": row["difficulty"], "template_id": row["template_id"],
            "context_size": row["context_size"], "split": row["split"], "method": row["method"],
            "condition": row["condition"], "status": row["status"], "error_stage": error.get("stage"),
            "error_message": error.get("message"), "exact_match": metrics.get("exact_match"),
            "strict_match": metrics.get("strict_match"), "precision": metrics.get("precision"),
            "recall": metrics.get("recall"), "f1": metrics.get("f1"), "predicted_count": metrics.get("predicted_count"),
            "gold_count": metrics.get("gold_count"), "duplicate_count": metrics.get("duplicate_count"),
            "invalid_count": len(metrics.get("invalid_ids", [])) if metrics else None,
            "input_tokens": row["tokens"]["input"], "output_tokens": row["tokens"]["output"],
            "engine_latency_seconds": row["timing"]["engine_latency_seconds"],
            "postprocessing_seconds": row["timing"]["postprocessing_seconds"]}


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    temporary = path.with_name("." + path.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def finalize(context: RunContext, run_dir: Path, run_metadata: dict, tasks: list[tuple[dict, str]]) -> dict:
    order = {result_key(example, condition): position for position, (example, condition) in enumerate(tasks)}
    results = sorted((row for key, row in context.journal.results.items() if key in order), key=lambda row: order[row["result_key"]])
    batches = list(context.journal.events("batch"))
    summary = summarize(results, context.method.conditions, batches)
    atomic_write_json(run_dir / "summary.json", summary)
    summary_columns = ["group_by", "group_value", "condition", "results", "scored", "input_errors", "correct", "accuracy",
                       "strict_accuracy", "precision", "recall", "f1", "ok_rate", "input_tokens", "output_tokens",
                       "token_counts_complete", "engine_latency_available", "engine_latency_seconds_mean",
                       "engine_latency_seconds_p95", "status_counts"]
    write_csv(run_dir / "summary.csv", summary_columns, summary["table"])
    write_csv(run_dir / "results.csv", CSV_COLUMNS, [flat_result(row) for row in results])
    run_metadata.update(completed_results=len(results), expected_results=len(tasks),
                        runtime_sessions=list(context.journal.events("runtime")),
                        failures=list(context.journal.events("fatal")), finished_at=utc_now())
    metadata = {"method": context.method.name, "conditions": list(context.method.conditions), "status": run_metadata["status"],
                "run": {key: value for key, value in run_metadata.items() if key not in {"runtime_sessions"}},
                "runtime_sessions": run_metadata["runtime_sessions"],
                "summary_overall": [row for row in summary["table"] if row["group_by"] == "overall"],
                "paired_comparison": summary["paired_comparison"], "limitations": LIMITATIONS}
    write_envelope(run_dir / "results.json", "retrieval_results", metadata, results)
    atomic_write_json(run_dir / "run_metadata.json", run_metadata)
    return summary


def default_run_dir(input_path: Path, method: str, args: argparse.Namespace) -> Path:
    leaf = args.split + (f"_perlevel{args.per_level}" if args.per_level else "") + ("_fake" if args.backend == "fake" else "")
    return ARTIFACTS_DIR / "results" / input_path.stem / method / leaf


def run_benchmark(method: Method, args: argparse.Namespace, logger: logging.Logger) -> int:
    if args.self_test:
        return self_test_harness(method, args, logger)
    input_path = args.input.resolve()
    run_dir = (args.run_dir or default_run_dir(input_path, method.name, args)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, run_dir / "run.log")
    with exclusive_lock(run_dir / "run"):
        benchmark = load_benchmark(input_path, args.split, args.per_level, args.seed, logger)
        context = RunContext(args=args, method=method, benchmark=benchmark, journal=None, logger=logger)
        method.setup(context)
        scripts = {Path(path).name: file_sha256(path) for path in
                   (method.script, Path(__file__), Path(__file__).with_name("pipeline_common.py"))}
        config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()
                  if key not in {"input", "run_dir", "resume", "log_level", "self_test", "smoke_test"}}
        parts = {"benchmark_sha256": benchmark.file_sha256, "scripts": scripts, "prompts_sha256": stable_hash(method.prompts),
                 "config": config, **context.fingerprint_extra}
        fingerprint = stable_hash(parts)
        context.journal = journal = Journal(run_dir / "journal.jsonl", args.resume)
        try:
            if args.resume:
                require(journal.header["fingerprint"] == fingerprint,
                        "Resume rejected: benchmark, scripts, prompts, semantic model or configuration changed; use a new --run-dir")
            else:
                journal.append("header", {"created_at": utc_now(), "fingerprint": fingerprint, **parts,
                                          "benchmark_path": str(input_path), "domain": benchmark.domain,
                                          "selection": benchmark.selection, "prompts": method.prompts})
            examples = benchmark.examples
            tasks = [(example, condition) for position, example in enumerate(examples)
                     for condition in (method.conditions if position % 2 == 0 else method.conditions[::-1])]
            pending = [task for task in tasks if result_key(*task) not in journal.results]
            run_metadata = {"method": method.name, "run_dir": str(run_dir), "status": "running",
                            "benchmark": {"path": str(input_path), "file_sha256": benchmark.file_sha256,
                                          "domain": benchmark.domain, "complete": benchmark.metadata.get("complete"),
                                          "evidence_schema": benchmark.schema["name"]},
                            "selection": benchmark.selection, "fingerprint": fingerprint, "fingerprint_parts": parts,
                            "started_at": journal.header["created_at"], "session_started_at": utc_now()}
            atomic_write_json(run_dir / "run_metadata.json", run_metadata)
            logger.info("%s on %s (%s): %d examples x %d conditions = %d tasks, %d pending; run dir %s", method.name,
                        input_path.name, benchmark.domain, len(examples), len(method.conditions), len(tasks), len(pending), run_dir)

            def on_sigterm(signum, frame):
                raise Terminated(f"signal {signum}")
            previous = signal.signal(signal.SIGTERM, on_sigterm)
            code = EXIT_OK
            try:
                for start in range(0, len(pending), args.batch_size):
                    batch = pending[start:start + args.batch_size]
                    for row in method.process(context, batch):
                        journal.append("result", row)
                    logger.info("Results %d/%d", len([key for key in journal.results if key in {result_key(*t) for t in tasks}]), len(tasks))
                run_metadata["status"] = "complete"
            except (KeyboardInterrupt, Terminated) as error:
                logger.warning("Interrupted (%s); rerun the same command with --resume", error)
                journal.append("fatal", {"type": type(error).__name__, "message": str(error)})
                run_metadata["status"], code = "interrupted", EXIT_INTERRUPTED
            except Exception as error:  # noqa: BLE001 - record, export partial results, then fail
                logger.exception("Run failed; partial results exported. Fix the cause, then --resume")
                journal.append("fatal", {"type": type(error).__name__, "message": str(error)})
                run_metadata["status"], code = "failed", EXIT_ERROR
            finally:
                signal.signal(signal.SIGTERM, previous)
                summary = finalize(context, run_dir, run_metadata, tasks)
            for row in summary["table"]:
                if row["group_by"] in {"overall", "difficulty"}:
                    logger.info("%-12s %-6s %-16s acc=%s (%d/%d) ok_rate=%s", row["group_by"], row["group_value"],
                                row["condition"], _fmt(row["accuracy"]), row["correct"], row["scored"], _fmt(row["ok_rate"]))
            return code
        finally:
            journal.close()


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


# ---------------------------------------------------------------------------
# Self-test: synthetic benchmarks for both domains, fake backend, resume, fingerprint
# ---------------------------------------------------------------------------
def synthetic_examples(domain: str) -> list[dict]:
    schema = EVIDENCE_SCHEMAS[domain]
    examples = []
    for level in range(1, 6):
        for variant in range(2):
            rows = []
            for index in range(6):
                day = f"2021-0{1 + index}-1{variant}"
                if domain == "broker":
                    content = {"date": day, "ticker": "TST", "firm": f"Firm{index % 2}",
                               "currentPriceTarget": 100.0 + index, "priorPriceTarget": 90.0 + index}
                else:
                    content = {"cik": "0000000001", "ticker": "TST", "entity_name": "Test Co", "taxonomy": "us-gaap",
                               "concept": "Revenues", "item": "revenue", "label": None, "unit": "USD", "start": "2020-01-01",
                               "end": "2020-12-31", "val": 1000 + index, "accn": f"0000000001-21-00000{index}",
                               "form": "10-K", "filed": day, "source_url": "https://www.sec.gov/"}
                rows.append(dict(evidence_id="e_" + stable_hash(content)[:12], **content))
            base = f"b_{domain}{level}{variant}"
            gold = sorted(row["evidence_id"] for row in rows[:level])
            examples.append({"example_id": f"{base}_6", "base_id": base, "domain": domain, "difficulty": level,
                             "template_id": f"L{level}_synthetic", "split": "test", "entity_ciks": ["0000000001"],
                             "question": f"Synthetic question {level}. Return only a JSON list of evidence_id values.",
                             "cutoff": "2021-12-31", "context_size": 6, "gold_count": len(gold),
                             "gold_fraction": len(gold) / 6, "adjacency_radius_days": 30, "evidence_list": rows,
                             "golden_evidence_ids": gold,
                             "distractor_ids": sorted(row["evidence_id"] for row in rows[level:]),
                             "parameters": {}, "golden_conditions": {}, "private_answer": {}, "snapshot_hash": "x",
                             "diagnostics": {}})
    examples[-1]["evidence_list"][0]["ticker"] = None  # one invalid example exercises input_error
    return examples


def self_test_harness(method: Method, args: argparse.Namespace, logger: logging.Logger) -> int:
    for raw, thinking, expected in [("[\"a\"]", False, ("[\"a\"]", False)), ("x</think>\n```json\n[]\n```", True, ("[]", True))]:
        require(final_answer_text(raw, thinking) == expected, "final_answer_text changed")
    try:
        final_answer_text("<think>unfinished", False)
    except ValueError:
        pass
    else:
        raise PipelineError("Unfinished reasoning accepted")
    scored = score_predictions(["a", "a", "zz"], ["a"], ["a", "b"], True)
    require(not scored["exact_match"] and scored["duplicate_count"] == 1 and scored["invalid_ids"] == ["zz"], "Scoring changed")
    require(score_predictions(["a"], ["a"], ["a"], True)["strict_match"], "Strict match changed")
    require(score_predictions(["a"], ["a"], ["a"], False)["f1"] == 0.0, "Failure scoring changed")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for domain in ("broker", "edgar"):
            path = root / f"{domain}.json"
            metadata = {"domain": domain, "complete": True, "evidence_schema": EVIDENCE_SCHEMAS[domain]}
            examples = synthetic_examples(domain)
            write_envelope(path, "benchmark", metadata, examples)
            test_args = copy.copy(args)
            test_args.__dict__.update(input=path, run_dir=root / domain / method.name, self_test=False, resume=False,
                                      backend="fake", split="test", per_level=None, batch_size=3)
            require(run_benchmark(method, test_args, logger) == EXIT_OK, f"{domain} fake run failed")
            reader, results = read_envelope(test_args.run_dir / "results.json", "retrieval_results")
            require(len(results) == len(examples) * len(method.conditions), "Result count mismatch")
            require(sum(row["status"] == "input_error" for row in results) == len(method.conditions), "input_error path not exercised")
            require({row["status"] for row in results} - {"input_error"}, "No model results")
            test_args.resume = True
            require(run_benchmark(method, test_args, logger) == EXIT_OK, "Resume of a finished run failed")
            require(len(read_envelope(test_args.run_dir / "results.json")[1]) == len(results), "Resume duplicated results")
            test_args.seed += 1
            try:
                run_benchmark(method, test_args, logger)
            except PipelineError:
                pass
            else:
                raise PipelineError("Changed configuration was accepted on resume")
            test_args.__dict__.update(resume=False, seed=args.seed, per_level=1, run_dir=root / domain / "sample")
            require(run_benchmark(method, test_args, logger) == EXIT_OK, "Sampled run failed")
            require(len(read_envelope(test_args.run_dir / "results.json")[1]) == 5 * len(method.conditions), "Sampling wrong")
    setup_logging(args.log_level)
    logger.info("%s harness self-test passed for both domains", method.name)
    return EXIT_OK
