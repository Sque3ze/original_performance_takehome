#!/usr/bin/env python3
"""Sweep kernel environment knobs and report cycles."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from itertools import product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from perf_takehome import KernelBuilder, compute_slot_stats, print_slot_stats


def parse_values(spec: str):
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Bad sweep range: {spec}")
        start = int(parts[0])
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step == 0:
            raise ValueError(f"Bad sweep step: {spec}")
        if start <= end:
            return list(range(start, end + 1, step))
        return list(range(start, end - 1, -abs(step)))
    return [int(v) if v.strip().lstrip("-").isdigit() else v for v in spec.split(",")]


def apply_env(env: dict[str, str]):
    for key, value in env.items():
        os.environ[key] = str(value)


def run_cycles(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int | None,
    quiet: bool,
):
    import importlib.util
    import random

    tests_dir = os.path.join(ROOT, "tests")
    submission_path = os.path.join(tests_dir, "submission_tests.py")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)

    spec = importlib.util.spec_from_file_location("submission_tests", submission_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load submission_tests.py")
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)

    if seed is not None:
        random.seed(seed)

    st.kernel_builder.cache_clear()
    st.cycles.cache_clear()

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            return st.do_kernel_test(forest_height, rounds, batch_size)
    return st.do_kernel_test(forest_height, rounds, batch_size)


def build_stats(forest_height: int, rounds: int, batch_size: int):
    n_nodes = 2 ** (forest_height + 1) - 1
    kb = KernelBuilder()
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
    return compute_slot_stats(kb.instrs)


def main():
    parser = argparse.ArgumentParser(description="Sweep kernel config knobs.")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run output.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Set an environment variable for all runs.",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="KEY=VALS",
        help="Sweep values: e.g. WAVE_SIZE=64,72,80 or SHALLOW_GROUP_DEPTH=0:4",
    )
    parser.add_argument("--top", type=int, default=5, help="Show top-N results.")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print slot stats for the best run.",
    )
    parser.add_argument(
        "--stats-each",
        action="store_true",
        help="Print slot stats for every run.",
    )
    args = parser.parse_args()

    base_env = {}
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Bad --set: {item}")
        key, value = item.split("=", 1)
        base_env[key] = value

    sweep_env = []
    for item in args.sweep:
        if "=" not in item:
            raise ValueError(f"Bad --sweep: {item}")
        key, spec = item.split("=", 1)
        sweep_env.append((key, parse_values(spec)))

    if not sweep_env:
        sweep_env = [("WAVE_SIZE", [os.getenv("WAVE_SIZE", "82")])]

    results = []
    keys = [k for k, _ in sweep_env]
    values = [v for _, v in sweep_env]

    original_env = {k: os.environ.get(k) for k in set(base_env) | set(keys)}

    try:
        for combo in product(*values):
            env = dict(base_env)
            env.update({k: v for k, v in zip(keys, combo)})
            apply_env(env)
            try:
                cycles = run_cycles(
                    args.forest_height,
                    args.rounds,
                    args.batch_size,
                    args.seed,
                    args.quiet,
                )
            except AssertionError:
                cycles = float("inf")
            stats = None
            if args.stats_each:
                stats = build_stats(args.forest_height, args.rounds, args.batch_size)
            results.append((cycles, env, stats))
    finally:
        for k, v in original_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    results.sort(key=lambda x: x[0])

    print("\nResults:")
    for cycles, env, stats in results[: args.top]:
        cfg = " ".join(f"{k}={env[k]}" for k in sorted(env))
        print(f"- cycles={cycles:8.1f} {cfg}")
        if stats is not None:
            print_slot_stats(stats)

    if args.stats and results:
        best_cycles, best_env, _ = results[0]
        apply_env(best_env)
        print(f"\nBest config cycles={best_cycles:8.1f}")
        print_slot_stats(build_stats(args.forest_height, args.rounds, args.batch_size))


if __name__ == "__main__":
    main()
