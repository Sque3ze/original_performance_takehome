#!/usr/bin/env python3
"""Analyze instruction mix and engine utilization for the current kernel."""

from collections import defaultdict
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from perf_takehome import KernelBuilder, compute_slot_stats, print_slot_stats


def analyze(forest_height: int, rounds: int, batch_size: int):
    n_nodes = 2 ** (forest_height + 1) - 1
    kb = KernelBuilder()
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)

    print_slot_stats(compute_slot_stats(kb.instrs))


if __name__ == "__main__":
    analyze(forest_height=10, rounds=16, batch_size=256)
