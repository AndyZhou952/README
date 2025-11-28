"""Performance profiling utilities for vLLM-Omni pipeline.

This module provides functions to run multiple inference rounds, collect statistics,
and generate formatted performance summaries.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# Stage name mapping (assuming standard Qwen2.5-Omni pipeline)
STAGE_NAMES = {
    0: "THINKER",
    1: "TALKER",
    2: "CODE2WAV",
}


def parse_stats_jsonl(filepath: str) -> list[dict[str, Any]]:
    """Parse a JSONL stats file and return list of records.

    Args:
        filepath: Path to the JSONL stats file

    Returns:
        List of parsed JSON records
    """
    records = []
    if not os.path.exists(filepath):
        # Don't warn on initial check - file may not exist yet
        logger.debug("Stats file not found (may not exist yet): %s", filepath)
        return records

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse JSONL line in %s: %s", filepath, e)
    except Exception as e:
        logger.error("Failed to read stats file %s: %s", filepath, e)

    return records


def extract_orchestrator_summary(stats_file: str) -> dict[str, Any] | None:
    """Extract the latest orchestrator summary from stats file.

    Args:
        stats_file: Path to orchestrator stats JSONL file

    Returns:
        The latest orchestrator summary dict, or None if not found
    """
    records = parse_stats_jsonl(stats_file)
    # Find the last summary record (most recent)
    summary = None
    for record in records:
        if record.get("type") == "orchestrator_summary":
            summary = record
    return summary


def aggregate_profile_runs(run_stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics across multiple profiling runs.

    Args:
        run_stats_list: List of orchestrator summary dicts from each run

    Returns:
        Aggregated statistics dictionary
    """
    if not run_stats_list:
        return {}

    num_runs = len(run_stats_list)

    # Aggregate per-stage metrics
    stage_aggregates: dict[int, dict[str, Any]] = {}
    for run_stats in run_stats_list:
        stages = run_stats.get("stages", [])
        for stage_data in stages:
            stage_id = stage_data.get("stage_id", -1)
            if stage_id < 0:
                continue

            if stage_id not in stage_aggregates:
                stage_aggregates[stage_id] = {
                    "requests": 0,
                    "tokens": 0,
                    "total_time_ms": 0.0,
                    "avg_time_per_request_ms": 0.0,
                    "avg_tokens_per_s": 0.0,
                }

            agg = stage_aggregates[stage_id]
            agg["requests"] += stage_data.get("requests", 0)
            agg["tokens"] += stage_data.get("tokens", 0)
            agg["total_time_ms"] += stage_data.get("total_time_ms", 0.0)

    # Calculate averages for stages
    for stage_id, agg in stage_aggregates.items():
        if agg["requests"] > 0:
            agg["avg_time_per_request_ms"] = agg["total_time_ms"] / agg["requests"]
        if agg["total_time_ms"] > 0:
            agg["avg_tokens_per_s"] = (agg["tokens"] * 1000.0) / agg["total_time_ms"]

    # Aggregate transfer metrics
    transfer_aggregates: dict[tuple[int, int], dict[str, Any]] = {}
    for run_stats in run_stats_list:
        transfers = run_stats.get("transfers", [])
        for transfer_data in transfers:
            from_stage = transfer_data.get("from_stage", -1)
            to_stage = transfer_data.get("to_stage", -1)
            if from_stage < 0 or to_stage < 0:
                continue

            key = (from_stage, to_stage)
            if key not in transfer_aggregates:
                transfer_aggregates[key] = {
                    "samples": 0,
                    "total_bytes": 0,
                    "tx_time_ms": 0.0,
                    "tx_mbps": 0.0,
                    "rx_decode_time_ms": 0.0,
                    "total_transfer_time_ms": 0.0,
                }

            agg = transfer_aggregates[key]
            agg["samples"] += transfer_data.get("samples", 0)
            agg["total_bytes"] += transfer_data.get("total_bytes", 0)
            # total_time_ms is the TX time (sum_ms from transfer_agg)
            agg["tx_time_ms"] += transfer_data.get("total_time_ms", 0.0)
            # rx_total_time_ms is the RX decode time
            agg["rx_decode_time_ms"] += transfer_data.get("rx_total_time_ms", 0.0)
            # total_transfer_time_ms includes tx + rx + in_flight
            agg["total_transfer_time_ms"] += transfer_data.get("total_transfer_time_ms", 0.0)

    # Recalculate transfer metrics (averages)
    for key, agg in transfer_aggregates.items():
        if agg["samples"] > 0:
            avg_tx_time = agg["tx_time_ms"] / agg["samples"]
            if avg_tx_time > 0:
                agg["tx_mbps"] = (agg["total_bytes"] * 8.0) / (avg_tx_time * 1000.0)

    # Aggregate E2E metrics
    e2e_aggregates = {
        "total_requests": 0,
        "total_tokens": 0,
        "wall_time_ms": 0.0,
        "sum_e2e_time_ms": 0.0,
        "e2e_avg_time_per_request_ms": 0.0,
        "e2e_avg_tokens_per_s": 0.0,
    }

    for run_stats in run_stats_list:
        e2e_aggregates["total_requests"] += run_stats.get("e2e_requests", 0)
        e2e_aggregates["total_tokens"] += run_stats.get("e2e_total_tokens", 0)
        e2e_aggregates["wall_time_ms"] += run_stats.get("wall_time_ms", 0.0)
        e2e_aggregates["sum_e2e_time_ms"] += run_stats.get("e2e_sum_time_ms", 0.0)

    if e2e_aggregates["total_requests"] > 0:
        e2e_aggregates["e2e_avg_time_per_request_ms"] = (
            e2e_aggregates["sum_e2e_time_ms"] / e2e_aggregates["total_requests"]
        )

    if e2e_aggregates["sum_e2e_time_ms"] > 0:
        e2e_aggregates["e2e_avg_tokens_per_s"] = (
            e2e_aggregates["total_tokens"] * 1000.0
        ) / e2e_aggregates["sum_e2e_time_ms"]

    # Calculate pipeline efficiency
    pipeline_efficiency = 0.0
    if e2e_aggregates["wall_time_ms"] > 0:
        pipeline_efficiency = (
            e2e_aggregates["sum_e2e_time_ms"] / e2e_aggregates["wall_time_ms"]
        ) * 100.0

    return {
        "stages": stage_aggregates,
        "transfers": transfer_aggregates,
        "e2e": e2e_aggregates,
        "pipeline_efficiency": pipeline_efficiency,
        "num_runs": num_runs,
    }


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human-readable string."""
    if bytes_val < 1024:
        return f"{bytes_val} bytes"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.2f} KB"
    else:
        return f"{bytes_val / (1024 * 1024):.2f} MB"


def format_time_ms(ms: float) -> str:
    """Format milliseconds to human-readable string."""
    if ms < 1000:
        return f"{ms:.2f} ms"
    else:
        return f"{ms / 1000:.2f} s"


def format_performance_summary(aggregated_stats: dict[str, Any], num_runs: int) -> None:
    """Format and print the performance summary.

    Args:
        aggregated_stats: Aggregated statistics from aggregate_profile_runs
        num_runs: Number of profiling runs
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    # Per-stage metrics
    print("\nPER-STAGE METRICS:")
    print("-" * 70)

    stages = aggregated_stats.get("stages", {})
    for stage_id in sorted(stages.keys()):
        stage_data = stages[stage_id]
        stage_name = STAGE_NAMES.get(stage_id, f"STAGE-{stage_id}")

        print(f"\n{stage_name} (Stage {stage_id}):")
        print(f"  Requests processed: {stage_data.get('requests', 0)}")
        print(f"  Total tokens: {stage_data.get('tokens', 0)}")
        print(f"  Total time: {format_time_ms(stage_data.get('total_time_ms', 0.0))}")
        print(
            f"  Avg time/request: {format_time_ms(stage_data.get('avg_time_per_request_ms', 0.0))}"
        )
        print(
            f"  Avg throughput: {stage_data.get('avg_tokens_per_s', 0.0):.2f} tokens/s"
        )

    # Transfer metrics
    print("\nTRANSFER METRICS (between stages):")
    print("-" * 70)

    transfers = aggregated_stats.get("transfers", {})
    transfer_names = {
        (0, 1): "thinker -> talker",
        (1, 2): "talker -> code2wav",
    }

    for key in sorted(transfers.keys()):
        transfer_data = transfers[key]
        from_stage, to_stage = key
        transfer_name = transfer_names.get(key, f"stage-{from_stage} -> stage-{to_stage}")

        print(f"\n{transfer_name}:")
        print(f"  Transfer samples: {transfer_data.get('samples', 0)}")
        print(f"  Total bytes: {format_bytes(transfer_data.get('total_bytes', 0))}")
        print(f"  TX time: {format_time_ms(transfer_data.get('tx_time_ms', 0.0))}")
        print(f"  TX bandwidth: {transfer_data.get('tx_mbps', 0.0):.2f} Mbps")
        print(
            f"  RX decode time: {format_time_ms(transfer_data.get('rx_decode_time_ms', 0.0))}"
        )
        print(
            f"  Avg total transfer time: {format_time_ms(transfer_data.get('total_transfer_time_ms', 0.0))}"
        )

    # End-to-end metrics
    print("\nEND-TO-END METRICS:")
    print("-" * 70)

    e2e = aggregated_stats.get("e2e", {})
    pipeline_efficiency = aggregated_stats.get("pipeline_efficiency", 0.0)

    print(f"\n  Total requests: {e2e.get('total_requests', 0)}")
    print(f"  Total tokens: {e2e.get('total_tokens', 0)}")
    print(f"  Wall-clock time: {format_time_ms(e2e.get('wall_time_ms', 0.0))}")
    print(f"  Sum of E2E times: {format_time_ms(e2e.get('sum_e2e_time_ms', 0.0))}")
    print(
        f"  Avg E2E time/request: {format_time_ms(e2e.get('e2e_avg_time_per_request_ms', 0.0))}"
    )
    print(f"  Avg E2E throughput: {e2e.get('e2e_avg_tokens_per_s', 0.0):.2f} tokens/s")
    print(f"  Pipeline efficiency: {pipeline_efficiency:.1f}% (sum_e2e / wall_time)")

    print("\n" + "=" * 70)
    print(f"Profiling completed: {num_runs} runs")
    print("=" * 70 + "\n")


def extract_all_summaries(stats_file: str) -> list[dict[str, Any]]:
    """Extract all orchestrator summaries from stats file.

    Args:
        stats_file: Path to orchestrator stats JSONL file

    Returns:
        List of all orchestrator summary dicts
    """
    records = parse_stats_jsonl(stats_file)
    summaries = [r for r in records if r.get("type") == "orchestrator_summary"]
    return summaries


def run_profiling(
    omni_llm_factory: Any,
    prompts: list[dict[str, Any]],
    sampling_params_list: list[Any],
    num_runs: int,
    log_file: str | None,
) -> None:
    """Run profiling with multiple inference rounds.

    Args:
        omni_llm_factory: Callable that creates a new OmniLLM instance (since generate() closes it)
        prompts: List of prompts to use for inference
        sampling_params_list: List of sampling params for each stage
        num_runs: Number of profiling rounds to run
        log_file: Base log file path (stats will be written to {log_file}.orchestrator.stats.jsonl)
    """
    if log_file is None:
        log_file = "omni_llm_pipeline.log"

    stats_file = f"{log_file}.orchestrator.stats.jsonl"
    run_stats_list = []

    print(f"\n[Profiling] Starting {num_runs} profiling rounds...")
    print(f"[Profiling] Stats will be written to: {stats_file}\n")

    # Get initial count of summaries (in case file already exists)
    initial_summaries = extract_all_summaries(stats_file)
    initial_count = len(initial_summaries)

    for run_idx in range(num_runs):
        print(f"[Profiling] Run {run_idx + 1}/{num_runs}...")
        run_start_time = time.time()

        try:
            # Create a new OmniLLM instance for this run (since generate() closes it)
            # Note: This will remove old logs, so we need to extract summaries from previous runs first
            omni_llm = omni_llm_factory()
            # Run inference
            omni_outputs = omni_llm.generate(prompts, sampling_params_list)

            # Explicit cleanup - ensure processes are terminated and memory is freed
            try:
                omni_llm.close()
                del omni_llm
            except Exception:
                pass

            # Force garbage collection to free memory
            import gc

            gc.collect()

            # Extract stats from this run IMMEDIATELY (before next instance removes the file)
            # Note: The stats are written by OrchestratorMetrics.build_and_log_summary()
            # We need to wait a bit for the file to be written, then read it
            time.sleep(0.5)  # Small delay to ensure file is written

            # Read the latest summary (should be the one from this run)
            all_summaries = extract_all_summaries(stats_file)
            if all_summaries:
                # Use the last summary (most recent)
                summary = all_summaries[-1]
                run_stats_list.append(summary)
                run_time = time.time() - run_start_time
                print(f"[Profiling] Run {run_idx + 1} completed in {run_time:.2f}s")
            else:
                logger.warning(
                    "[Profiling] Could not extract summary for run %d", run_idx + 1
                )

            # Add delay between runs to allow process cleanup (if multiple runs)
            if run_idx < num_runs - 1:
                time.sleep(2.0)  # Give processes time to fully terminate

        except Exception as e:
            logger.error("[Profiling] Run %d failed: %s", run_idx + 1, e, exc_info=True)
            continue

    if not run_stats_list:
        print("\n[Profiling] ERROR: No valid stats collected from any run!")
        return

    # Aggregate and format results
    print("\n[Profiling] Aggregating results...")
    aggregated_stats = aggregate_profile_runs(run_stats_list)
    format_performance_summary(aggregated_stats, len(run_stats_list))

