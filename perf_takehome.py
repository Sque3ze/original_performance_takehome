"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
from contextlib import contextmanager
import os
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.labels = {}
        self.fixups = []
        self.sched_diag = None
        self.tag_enabled = os.getenv("TAG_DIAG", "0") == "1"
        self._tag = None

    @contextmanager
    def tag_scope(self, name):
        old = self._tag
        self._tag = name
        try:
            yield
        finally:
            self._tag = old

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot, tag=None):
        # Small, safe VLIW packing: only pack adjacent load-const ops.
        if engine == "load" and slot and slot[0] == "const" and self.instrs:
            last = self.instrs[-1]
            if (
                "load" in last
                and set(last.keys()) <= {"load", "debug"}
                and len(last["load"]) < SLOT_LIMITS["load"]
                and all(isinstance(s, tuple) and s and s[0] == "const" for s in last["load"])
            ):
                last["load"].append(slot)
                tag_name = tag if tag is not None else self._tag
                if self.tag_enabled and tag_name:
                    last.setdefault("debug", []).append(("tag", tag_name, engine))
                return

        bundle = {engine: [slot]}
        tag_name = tag if tag is not None else self._tag
        if self.tag_enabled and tag_name:
            bundle.setdefault("debug", []).append(("tag", tag_name, engine))
        self.instrs.append(bundle)

    def label(self, name):
        self.labels[name] = len(self.instrs)

    def add_jump(self, label):
        instr = {"flow": [("jump", 0)]}
        self.fixups.append((len(self.instrs), label, "abs"))
        self.instrs.append(instr)

    def add_cjump_rel(self, cond, label):
        instr = {"flow": [("cond_jump_rel", cond, 0)]}
        self.fixups.append((len(self.instrs), label, "rel"))
        self.instrs.append(instr)

    def resolve_fixups(self):
        for idx, label, kind in self.fixups:
            if label not in self.labels:
                raise ValueError(f"Unknown label {label}")
            target = self.labels[label]
            instr = self.instrs[idx]
            if kind == "abs":
                instr["flow"] = [("jump", target)]
            else:
                offset = target - (idx + 1)
                instr["flow"] = [("cond_jump_rel", instr["flow"][0][1], offset)]

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def emit_cycle(self, slots: list[tuple[Engine, tuple]], tag=None):
        if not slots:
            return
        bundle = defaultdict(list)
        counts = defaultdict(int)
        for engine, slot in slots:
            counts[engine] += 1
            assert counts[engine] <= SLOT_LIMITS[engine], f"{engine} slots exceeded"
            bundle[engine].append(slot)
            tag_name = tag if tag is not None else self._tag
            if self.tag_enabled and tag_name and engine != "debug":
                bundle["debug"].append(("tag", tag_name, engine))
        self.instrs.append(dict(bundle))

    def emit_vbroadcasts(self, pairs: list[tuple[int, int]]):
        limit = SLOT_LIMITS["valu"]
        for i in range(0, len(pairs), limit):
            chunk = pairs[i : i + limit]
            slots = [("valu", ("vbroadcast", dest, src)) for dest, src in chunk]
            self.emit_cycle(slots)

    def emit_load_offsets(self, dest_base: int, addr_base: int):
        limit = SLOT_LIMITS["load"]
        for lane in range(0, VLEN, limit):
            slots = []
            for i in range(limit):
                if lane + i >= VLEN:
                    break
                slots.append(
                    ("load", ("load_offset", dest_base, addr_base, lane + i))
                )
            self.emit_cycle(slots)

    def _slot_reads_writes(self, engine, slot):
        reads = set()
        writes = set()
        if engine == "alu":
            _, dest, a1, a2 = slot
            writes.add(dest)
            reads.add(a1)
            reads.add(a2)
        elif engine == "valu":
            match slot:
                case ("vbroadcast", dest, src):
                    reads.add(src)
                    writes.update(range(dest, dest + VLEN))
                case ("multiply_add", dest, a, b, c):
                    reads.update(range(a, a + VLEN))
                    reads.update(range(b, b + VLEN))
                    reads.update(range(c, c + VLEN))
                    writes.update(range(dest, dest + VLEN))
                case (op, dest, a1, a2):
                    reads.update(range(a1, a1 + VLEN))
                    reads.update(range(a2, a2 + VLEN))
                    writes.update(range(dest, dest + VLEN))
                case _:
                    return None, None
        elif engine == "load":
            match slot:
                case ("load", dest, addr):
                    reads.add(addr)
                    writes.add(dest)
                case ("load_offset", dest, addr, offset):
                    reads.add(addr + offset)
                    writes.add(dest + offset)
                case ("vload", dest, addr):
                    reads.add(addr)
                    writes.update(range(dest, dest + VLEN))
                case ("const", dest, _val):
                    writes.add(dest)
                case _:
                    return None, None
        elif engine == "store":
            match slot:
                case ("store", addr, src):
                    reads.add(addr)
                    reads.add(src)
                case ("vstore", addr, src):
                    reads.add(addr)
                    reads.update(range(src, src + VLEN))
                case _:
                    return None, None
        elif engine == "flow":
            match slot:
                case ("select", dest, cond, a, b):
                    reads.update([cond, a, b])
                    writes.add(dest)
                case ("add_imm", dest, a, _imm):
                    reads.add(a)
                    writes.add(dest)
                case ("vselect", dest, cond, a, b):
                    reads.update(range(cond, cond + VLEN))
                    reads.update(range(a, a + VLEN))
                    reads.update(range(b, b + VLEN))
                    writes.update(range(dest, dest + VLEN))
                case _:
                    return None, None
        else:
            return None, None
        return reads, writes

    def schedule_wave(self, ops_by_block: list[list[tuple[Engine, tuple]]], rr_start=0):
        n_blocks = len(ops_by_block)
        rr = rr_start % n_blocks if n_blocks else 0
        rr_mode = os.getenv("SCHED_RR_MODE", "round_robin")
        rr_load = rr
        load_chunk_cycles = int(os.getenv("SCHED_LOAD_CHUNK_CYCLES", "2"))
        if load_chunk_cycles < 1:
            load_chunk_cycles = 1
        load_chunk_count = 0
        window = int(os.getenv("SCHED_WINDOW", "2"))
        allow_multi = os.getenv("SCHED_MULTI_ISSUE", "1") == "1"
        resource_first = os.getenv("SCHED_RESOURCE_FIRST", "1") == "1"
        load_select = os.getenv("SCHED_LOAD_SELECT", "rr")
        if load_select not in ("rr", "ready_first", "earliest", "tag_first"):
            load_select = "rr"
        load_tag_set = {
            t.strip()
            for t in os.getenv("SCHED_LOAD_TAGS", "gather:load").split(",")
            if t.strip()
        }
        bubble_recovery = os.getenv("SCHED_BUBBLE_RECOVERY", "0") == "1"
        bubble_cycles = int(os.getenv("SCHED_BUBBLE_RECOVERY_CYCLES", "4"))
        if bubble_cycles < 1:
            bubble_cycles = 1
        bubble_engines = {
            t.strip()
            for t in os.getenv("SCHED_BUBBLE_RECOVERY_ENGINES", "alu,valu").split(",")
            if t.strip()
        }
        bubble_tag_set = {
            t.strip()
            for t in os.getenv("SCHED_BUBBLE_TAGS", "gather:addr,gather:load").split(",")
            if t.strip()
        }
        bubble_hold_rr = os.getenv("SCHED_BUBBLE_RECOVERY_HOLD_RR", "1") == "1"
        bubble_budget = 0
        enable_load_recovery = os.getenv("SCHED_LOAD_RECOVERY", "0") == "1"
        recovery_cycles = int(os.getenv("SCHED_LOAD_RECOVERY_CYCLES", "2"))
        if recovery_cycles < 1:
            recovery_cycles = 1
        recovery_engine = os.getenv("SCHED_LOAD_RECOVERY_ENGINE", "alu")
        if recovery_engine not in ("alu", "valu"):
            recovery_engine = "alu"
        recovery_tags = {
            t.strip()
            for t in os.getenv("SCHED_LOAD_RECOVERY_TAGS", "gather:addr").split(",")
            if t.strip()
        }
        recovery_budget = 0
        for b in range(n_blocks):
            ops = ops_by_block[b]
            if not ops:
                continue
            first_len = len(ops[0])
            if first_len in (2, 3):
                normalized = []
                for i, item in enumerate(ops):
                    if len(item) == 2:
                        engine, slot = item
                        normalized.append((i, engine, slot, None))
                    else:
                        if isinstance(item[0], int):
                            seq, engine, slot = item
                            normalized.append((seq, engine, slot, None))
                        else:
                            engine, slot, tag = item
                            normalized.append((i, engine, slot, tag))
                ops_by_block[b] = normalized
            elif first_len == 4:
                continue
        diag_enabled = os.getenv("SCHED_DIAG", "0") == "1"
        if diag_enabled and self.sched_diag is None:
            self.sched_diag = {
                "cycles": 0,
                "no_candidate": defaultdict(int),
                "rejects": defaultdict(int),
                "ready": defaultdict(int),
                "issued": defaultdict(int),
                "underfill_ready": defaultdict(int),
            }

        def _choose_candidate(ops, engine, block_sched, block_issues):
            prior_reads = set()
            prior_writes = set()
            chosen = None
            chosen_reads = None
            chosen_writes = None
            chosen_unknown = False

            for oi, (seq, op_engine, slot, tag) in enumerate(ops[:window]):
                reads, writes = self._slot_reads_writes(op_engine, slot)
                if reads is None or writes is None:
                    # Unknown slot shape; treat as a barrier.
                    if oi == 0 and not block_issues and op_engine == engine:
                        chosen = (seq, op_engine, slot, oi, tag)
                        chosen_unknown = True
                    elif diag_enabled and op_engine == engine:
                        self.sched_diag["rejects"]["barrier"] += 1
                    break

                # Can't move this op ahead of earlier unscheduled ops if it would
                # violate RAW/WAW/WAR with those earlier ops.
                if reads & prior_writes or writes & prior_writes or writes & prior_reads:
                    if diag_enabled and op_engine == engine:
                        if reads & prior_writes:
                            self.sched_diag["rejects"]["raw"] += 1
                        if writes & prior_writes:
                            self.sched_diag["rejects"]["waw"] += 1
                        if writes & prior_reads:
                            self.sched_diag["rejects"]["war"] += 1
                    prior_reads.update(reads)
                    prior_writes.update(writes)
                    continue

                if op_engine != engine:
                    prior_reads.update(reads)
                    prior_writes.update(writes)
                    continue

                # Same-cycle hazards within the block.
                conflict = False
                for sched_seq, sched_reads, sched_writes in block_sched:
                    if writes & sched_writes:
                        conflict = True
                        break
                    if reads & sched_writes and sched_seq < seq:
                        conflict = True
                        break
                if conflict:
                    if diag_enabled and op_engine == engine:
                        self.sched_diag["rejects"]["cycle_conflict"] += 1
                    prior_reads.update(reads)
                    prior_writes.update(writes)
                    continue

                chosen = (seq, op_engine, slot, oi, tag)
                chosen_reads = reads
                chosen_writes = writes
                break

            if chosen is None:
                return None
            return chosen, chosen_reads, chosen_writes, chosen_unknown

        remaining_slots = defaultdict(int)
        for ops in ops_by_block:
            for _seq, engine, _slot, _tag in ops:
                remaining_slots[engine] += 1
        while any(ops_by_block[b] for b in range(n_blocks)):
            bundle = defaultdict(list)
            counts = defaultdict(int)
            block_issues = [0] * n_blocks
            block_single = [False] * n_blocks
            block_sched = [[] for _ in range(n_blocks)]
            bubble_active = bubble_recovery and bubble_budget > 0
            bubble_order = None
            if bubble_active and n_blocks:
                block_load_dist = [10**9] * n_blocks
                block_tag_dist = [10**9] * n_blocks
                for b in range(n_blocks):
                    ops = ops_by_block[b]
                    if not ops:
                        continue
                    for oi, (_seq, op_engine, _slot, _tag) in enumerate(ops):
                        if _tag in bubble_tag_set and block_tag_dist[b] == 10**9:
                            block_tag_dist[b] = oi
                        if op_engine == "load":
                            block_load_dist[b] = oi
                            break
                bubble_order = sorted(
                    [b for b in range(n_blocks) if ops_by_block[b]],
                    key=lambda b: (
                        block_tag_dist[b],
                        block_load_dist[b],
                        (b - rr) % n_blocks,
                    ),
                )
            if resource_first:
                engine_order = ["load", "valu", "alu", "store", "flow"]
                priority = [
                    e
                    for e in engine_order
                    if e in SLOT_LIMITS and remaining_slots.get(e, 0) > 0
                ]
            else:
                priority = sorted(
                    [
                        e
                        for e in SLOT_LIMITS.keys()
                        if e != "debug" and remaining_slots.get(e, 0) > 0
                    ],
                    key=lambda e: remaining_slots.get(e, 0) / SLOT_LIMITS[e],
                    reverse=True,
                )
            if enable_load_recovery and recovery_budget > 0 and recovery_engine in priority:
                priority = [recovery_engine] + [e for e in priority if e != recovery_engine]
            for engine in priority:
                limit = SLOT_LIMITS[engine]
                while counts[engine] < limit:
                    scheduled_any = False
                    rr_engine = rr
                    if engine == "load" and rr_mode in ("sticky_load", "chunked_load"):
                        rr_engine = rr_load

                    if engine == "load" and load_select != "rr":
                        best_choice = None
                        best_key = None
                        found_front = False
                        for bi in range(n_blocks):
                            b = (rr_engine + bi) % n_blocks
                            ops = ops_by_block[b]
                            if not ops:
                                continue
                            if block_single[b] and block_issues[b]:
                                continue
                            if not allow_multi and block_issues[b]:
                                continue

                            cand = _choose_candidate(
                                ops, engine, block_sched[b], block_issues[b]
                            )
                            if cand is None:
                                continue
                            chosen, chosen_reads, chosen_writes, chosen_unknown = cand
                            oi = chosen[3]
                            if load_select == "ready_first":
                                if oi == 0:
                                    if not found_front:
                                        best_choice = (b, cand)
                                        best_key = bi
                                        found_front = True
                                    continue
                                if found_front:
                                    continue
                                if best_choice is None:
                                    best_choice = (b, cand)
                                    best_key = bi
                            elif load_select == "tag_first":
                                tag = chosen[4]
                                key = (0 if tag in load_tag_set else 1, oi, bi)
                                if best_key is None or key < best_key:
                                    best_key = key
                                    best_choice = (b, cand)
                            else:
                                key = (oi, bi)
                                if best_key is None or key < best_key:
                                    best_key = key
                                    best_choice = (b, cand)

                        if best_choice is not None:
                            b, cand = best_choice
                            chosen, chosen_reads, chosen_writes, chosen_unknown = cand
                            seq, op_engine, slot, oi, tag = chosen
                            bundle[op_engine].append(slot)
                            if self.tag_enabled and tag:
                                bundle["debug"].append(("tag", tag, op_engine))
                            counts[op_engine] += 1
                            remaining_slots[op_engine] -= 1
                            if chosen_unknown:
                                block_single[b] = True
                            elif chosen_reads is not None and chosen_writes is not None:
                                block_sched[b].append((seq, chosen_reads, chosen_writes))
                            block_issues[b] += 1
                            ops_by_block[b].pop(oi)
                            scheduled_any = True
                            if counts[engine] >= limit:
                                break
                        if not scheduled_any:
                            break
                    else:
                        if bubble_active and bubble_order is not None and engine in bubble_engines:
                            block_iter = bubble_order
                        else:
                            block_iter = ((rr_engine + bi) % n_blocks for bi in range(n_blocks))
                        for b in block_iter:
                            ops = ops_by_block[b]
                            if not ops:
                                continue
                            if block_single[b] and block_issues[b]:
                                continue
                            if not allow_multi and block_issues[b]:
                                continue

                            cand = _choose_candidate(
                                ops, engine, block_sched[b], block_issues[b]
                            )
                            if cand is None:
                                continue
                            if (
                                enable_load_recovery
                                and recovery_budget > 0
                                and engine == recovery_engine
                            ):
                                tag = cand[0][4]
                                if tag not in recovery_tags:
                                    continue

                            chosen, chosen_reads, chosen_writes, chosen_unknown = cand
                            seq, op_engine, slot, oi, tag = chosen
                            bundle[op_engine].append(slot)
                            if self.tag_enabled and tag:
                                bundle["debug"].append(("tag", tag, op_engine))
                            counts[op_engine] += 1
                            remaining_slots[op_engine] -= 1
                            if chosen_unknown:
                                block_single[b] = True
                            elif chosen_reads is not None and chosen_writes is not None:
                                block_sched[b].append((seq, chosen_reads, chosen_writes))
                            block_issues[b] += 1
                            ops.pop(oi)
                            scheduled_any = True
                            if counts[engine] >= limit:
                                break

                    if not scheduled_any:
                        break
                if diag_enabled and counts[engine] < limit and remaining_slots.get(engine, 0) > 0:
                    self.sched_diag["no_candidate"][engine] += (limit - counts[engine])
                if diag_enabled:
                    ready = 0
                    for b in range(n_blocks):
                        if not ops_by_block[b]:
                            continue
                        if block_single[b] and block_issues[b]:
                            continue
                        if not allow_multi and block_issues[b]:
                            continue
                        ops = ops_by_block[b]
                        prior_reads = set()
                        prior_writes = set()
                        local_sched = list(block_sched[b])
                        for oi, (seq, op_engine, slot, _tag) in enumerate(ops[:window]):
                            reads, writes = self._slot_reads_writes(op_engine, slot)
                            if reads is None or writes is None:
                                if oi == 0 and not block_issues[b] and op_engine == engine:
                                    ready += 1
                                break
                            if (
                                reads & prior_writes
                                or writes & prior_writes
                                or writes & prior_reads
                            ):
                                prior_reads.update(reads)
                                prior_writes.update(writes)
                                continue
                            if op_engine != engine:
                                prior_reads.update(reads)
                                prior_writes.update(writes)
                                continue
                            conflict = False
                            for sched_seq, sched_reads, sched_writes in local_sched:
                                if writes & sched_writes:
                                    conflict = True
                                    break
                                if reads & sched_writes and sched_seq < seq:
                                    conflict = True
                                    break
                            if conflict:
                                prior_reads.update(reads)
                                prior_writes.update(writes)
                                continue
                            ready += 1
                            if not allow_multi:
                                break
                            local_sched.append((seq, reads, writes))
                            prior_reads.update(reads)
                            prior_writes.update(writes)
                    self.sched_diag["ready"][engine] += ready
                    self.sched_diag["issued"][engine] += counts[engine]
                    if ready > counts[engine] and counts[engine] < limit:
                        self.sched_diag["underfill_ready"][engine] += (
                            min(limit, ready) - counts[engine]
                        )

            if enable_load_recovery and counts.get("load", 0) < SLOT_LIMITS["load"]:
                recovery_budget = recovery_cycles
            elif recovery_budget > 0:
                recovery_budget -= 1
            if bubble_recovery and remaining_slots.get("load", 0) > 0:
                if counts.get("load", 0) < SLOT_LIMITS["load"]:
                    bubble_budget = bubble_cycles
                elif bubble_budget > 0:
                    bubble_budget -= 1
            elif bubble_budget > 0:
                bubble_budget -= 1
            if not bundle:
                break
            self.instrs.append(dict(bundle))
            if n_blocks:
                hold_rr = bubble_recovery and bubble_hold_rr and counts.get("load", 0) < SLOT_LIMITS["load"]
                if not hold_rr:
                    rr = (rr + 1) % n_blocks
                if rr_mode == "round_robin":
                    rr_load = rr
                else:
                    load_issued = counts.get("load", 0) > 0
                    if rr_mode == "sticky_load":
                        if not load_issued:
                            rr_load = (rr_load + 1) % n_blocks
                    elif rr_mode == "chunked_load":
                        if load_issued:
                            load_chunk_count += 1
                            if load_chunk_count >= load_chunk_cycles:
                                rr_load = (rr_load + 1) % n_blocks
                                load_chunk_count = 0
                        else:
                            rr_load = (rr_load + 1) % n_blocks
                            load_chunk_count = 0
            if diag_enabled:
                self.sched_diag["cycles"] += 1

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Vectorized implementation using SIMD with wave scheduling and shallow grouping.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        if init_vars:
            self.emit_cycle([("load", ("const", tmp1, 0))], tag="init:hdr")
            for i, v in enumerate(init_vars):
                slots = [("load", ("load", self.scratch[v], tmp1))]
                if i + 1 < len(init_vars):
                    slots.append(("load", ("const", tmp1, i + 1)))
                self.emit_cycle(slots, tag="init:hdr")

        extra_room_size = n_nodes + batch_size * 2 + VLEN * 2 + 32

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        blocks = batch_size // VLEN
        tail_start = blocks * VLEN

        group_depth = int(os.getenv("PARTITION_GROUP_DEPTH", "1"))
        partition_grouping = os.getenv("PARTITION_GROUPING", "0") == "1"
        block_offsets = [self.scratch_const(i * VLEN) for i in range(blocks)]

        # Vector constants
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_three = self.alloc_scratch("v_three", VLEN)
        v_four = self.alloc_scratch("v_four", VLEN)

        vbroadcasts = [
            (v_zero, zero_const),
            (v_one, one_const),
            (v_two, two_const),
            (v_three, self.scratch_const(3)),
            (v_four, self.scratch_const(4)),
        ]

        # Hash constants and fused stages
        v_m12 = self.alloc_scratch("v_m12", VLEN)
        v_m5 = self.alloc_scratch("v_m5", VLEN)
        v_m3 = self.alloc_scratch("v_m3", VLEN)
        vbroadcasts.extend(
            [
                (v_m12, self.scratch_const(4097)),
                (v_m5, self.scratch_const(33)),
                (v_m3, self.scratch_const(9)),
            ]
        )

        hash_plan = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.scratch_const(val1)
            v_c1 = self.alloc_scratch(f"v_c1_{hi}", VLEN)
            vbroadcasts.append((v_c1, c1))
            if hi == 0:
                hash_plan.append(("fused", v_m12, v_c1))
            elif hi == 2:
                hash_plan.append(("fused", v_m5, v_c1))
            elif hi == 4:
                hash_plan.append(("fused", v_m3, v_c1))
            else:
                c3 = self.scratch_const(val3)
                v_c3 = self.alloc_scratch(f"v_c3_{hi}", VLEN)
                vbroadcasts.append((v_c3, c3))
                hash_plan.append(("normal", op1, v_c1, op2, op3, v_c3))

        with self.tag_scope("preamble:vbroadcast"):
            self.emit_vbroadcasts(vbroadcasts)
        hash_scalar_consts = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            c1 = self.scratch_const(val1)
            c3 = self.scratch_const(val3)
            hash_scalar_consts.append((op1, c1, op2, op3, c3))

        use_flow_select = os.getenv("USE_FLOW_SELECT", "0") == "1"
        flow_select_depth1 = os.getenv("FLOW_SELECT_DEPTH1", "1") == "1"
        flow_select_depth2 = os.getenv("FLOW_SELECT_DEPTH2", "0") == "1"
        flow_select_depth3 = os.getenv("FLOW_SELECT_DEPTH3", "0") == "1"
        valu_tree_depth3 = os.getenv("VALU_TREE_DEPTH3", "0") == "1"
        if use_flow_select:
            flow_select_depth1 = True
            flow_select_depth2 = True
            flow_select_depth3 = True
        fast_depth2_select = os.getenv("FAST_DEPTH2_SELECT", "1") != "0"
        fast_depth3_select = os.getenv("FAST_DEPTH3_SELECT", "1") != "0"

        use_shallow_grouping = os.getenv("USE_SHALLOW_GROUPING", "1") != "0"
        shallow_group_depth = int(os.getenv("SHALLOW_GROUP_DEPTH", "2"))
        node_vals_s = []
        node_idx_consts = []
        depth_offsets = []
        v_node_off_all = None
        v_node_val_all = None
        v_depth1_diff = None
        v_depth2_diff10 = None
        v_depth2_diff20 = None
        v_depth2_diff30 = None
        v_depth3_diff10 = None
        v_depth3_diff20 = None
        v_depth3_diff30 = None
        v_depth3_diff40 = None
        v_depth3_diff50 = None
        v_depth3_diff60 = None
        v_depth3_diff70 = None
        need_node_off = False
        special_depth2_addr = False
        if use_shallow_grouping:
            if shallow_group_depth >= 2 and not (flow_select_depth2 or fast_depth2_select):
                need_node_off = True
            if shallow_group_depth >= 3 and not (
                fast_depth3_select or flow_select_depth3 or valu_tree_depth3
            ):
                need_node_off = True
            if shallow_group_depth >= 4:
                need_node_off = True
            special_depth2_addr = shallow_group_depth == 2 and not need_node_off

        if use_shallow_grouping:
            max_group_node = (1 << (shallow_group_depth + 1)) - 2
            total_group_nodes = sum(1 << d for d in range(shallow_group_depth + 1))
            node_vals_s = [
                self.alloc_scratch(f"node_val_s_{i}") for i in range(max_group_node + 1)
            ]
            if special_depth2_addr:
                node_idx_consts = [self.scratch_const(i) for i in range(5)]
            else:
                node_idx_consts = [self.scratch_const(i) for i in range(max_group_node + 1)]
            node_addr_s = [
                self.alloc_scratch(f"node_addr_s_{i}") for i in range(max_group_node + 1)
            ]
            offset = 0
            for depth in range(shallow_group_depth + 1):
                depth_offsets.append(offset)
                offset += (1 << depth) * VLEN
            if need_node_off:
                v_node_off_all = self.alloc_scratch(
                    "v_node_off_all", total_group_nodes * VLEN
                )
            v_node_val_all = self.alloc_scratch(
                "v_node_val_all", total_group_nodes * VLEN
            )
            addr_ops = []
            with self.tag_scope("preamble:group_addr"):
                for node_idx in range(max_group_node + 1):
                    if special_depth2_addr and node_idx >= len(node_idx_consts):
                        continue
                    addr_ops.append(
                        (
                            "alu",
                            (
                                "+",
                                node_addr_s[node_idx],
                                self.scratch["forest_values_p"],
                                node_idx_consts[node_idx],
                            ),
                        )
                    )
                    if len(addr_ops) == SLOT_LIMITS["alu"]:
                        self.emit_cycle(addr_ops)
                        addr_ops = []
                if addr_ops:
                    self.emit_cycle(addr_ops)
            with self.tag_scope("preamble:group_load"):
                for i in range(0, max_group_node + 1, SLOT_LIMITS["load"]):
                    slots = []
                    for j in range(i, min(max_group_node + 1, i + SLOT_LIMITS["load"])):
                        slots.append(("load", ("load", node_vals_s[j], node_addr_s[j])))
                    if special_depth2_addr and i == 0:
                        slots.append(
                            ("alu", ("+", node_addr_s[5], node_addr_s[4], one_const))
                        )
                        slots.append(
                            ("alu", ("+", node_addr_s[6], node_addr_s[4], two_const))
                        )
                    self.emit_cycle(slots)
            idx_pairs = []
            val_pairs = []
            for depth in range(shallow_group_depth + 1):
                base = (1 << depth) - 1
                num_nodes = 1 << depth
                depth_off = depth_offsets[depth]
                for node_k in range(num_nodes):
                    node_idx = base + node_k
                    if need_node_off:
                        idx_pairs.append(
                            (
                                v_node_off_all + depth_off + node_k * VLEN,
                                node_idx_consts[node_k],
                            )
                        )
                    val_pairs.append(
                        (
                            v_node_val_all + depth_off + node_k * VLEN,
                            node_vals_s[node_idx],
                        )
                    )
            if idx_pairs:
                with self.tag_scope("preamble:group_broadcast"):
                    self.emit_vbroadcasts(idx_pairs)
            with self.tag_scope("preamble:group_broadcast"):
                self.emit_vbroadcasts(val_pairs)
            diff_ops = []
            diff1_s = None
            diff2_10_s = None
            diff2_20_s = None
            diff2_30_s = None
            diff3_10_s = None
            diff3_20_s = None
            diff3_30_s = None
            diff3_40_s = None
            diff3_50_s = None
            diff3_60_s = None
            diff3_70_s = None
            if shallow_group_depth >= 1 and not flow_select_depth1:
                diff1_s = self.alloc_scratch("diff1_s")
                base1 = (1 << 1) - 1
                diff_ops.append(
                    (
                        "alu",
                        ("-", diff1_s, node_vals_s[base1 + 1], node_vals_s[base1 + 0]),
                    )
                )
            if shallow_group_depth >= 2 and fast_depth2_select and not flow_select_depth2:
                base2 = (1 << 2) - 1
                diff2_10_s = self.alloc_scratch("diff2_10_s")
                diff2_20_s = self.alloc_scratch("diff2_20_s")
                diff2_30_s = self.alloc_scratch("diff2_30_s")
                diff_ops.extend(
                    [
                        (
                            "alu",
                            ("-", diff2_10_s, node_vals_s[base2 + 1], node_vals_s[base2 + 0]),
                        ),
                        (
                            "alu",
                            ("-", diff2_20_s, node_vals_s[base2 + 2], node_vals_s[base2 + 0]),
                        ),
                        (
                            "alu",
                            ("-", diff2_30_s, node_vals_s[base2 + 3], node_vals_s[base2 + 0]),
                        ),
                    ]
                )
            if shallow_group_depth >= 3 and fast_depth3_select and not (
                flow_select_depth3 or valu_tree_depth3
            ):
                base3 = (1 << 3) - 1
                diff3_10_s = self.alloc_scratch("diff3_10_s")
                diff3_20_s = self.alloc_scratch("diff3_20_s")
                diff3_30_s = self.alloc_scratch("diff3_30_s")
                diff3_40_s = self.alloc_scratch("diff3_40_s")
                diff3_50_s = self.alloc_scratch("diff3_50_s")
                diff3_60_s = self.alloc_scratch("diff3_60_s")
                diff3_70_s = self.alloc_scratch("diff3_70_s")
                diff_ops.extend(
                    [
                        (
                            "alu",
                            ("-", diff3_10_s, node_vals_s[base3 + 1], node_vals_s[base3 + 0]),
                        ),
                        (
                            "alu",
                            ("-", diff3_20_s, node_vals_s[base3 + 2], node_vals_s[base3 + 0]),
                        ),
                        (
                            "alu",
                            ("-", diff3_30_s, node_vals_s[base3 + 3], node_vals_s[base3 + 0]),
                        ),
                        (
                            "alu",
                            ("-", diff3_40_s, node_vals_s[base3 + 4], node_vals_s[base3 + 0]),
                        ),
                        (
                            "alu",
                            ("-", diff3_50_s, node_vals_s[base3 + 5], node_vals_s[base3 + 0]),
                        ),
                        (
                            "alu",
                            ("-", diff3_60_s, node_vals_s[base3 + 6], node_vals_s[base3 + 0]),
                        ),
                        (
                            "alu",
                            ("-", diff3_70_s, node_vals_s[base3 + 7], node_vals_s[base3 + 0]),
                        ),
                    ]
                )
            if diff_ops:
                with self.tag_scope("preamble:group_diff"):
                    self.emit_cycle(diff_ops)
            if diff1_s is not None:
                v_depth1_diff = self.alloc_scratch("v_depth1_diff", VLEN)
                with self.tag_scope("preamble:group_diff"):
                    self.emit_cycle([("valu", ("vbroadcast", v_depth1_diff, diff1_s))])
            if diff2_10_s is not None:
                v_depth2_diff10 = self.alloc_scratch("v_depth2_diff10", VLEN)
                v_depth2_diff20 = self.alloc_scratch("v_depth2_diff20", VLEN)
                v_depth2_diff30 = self.alloc_scratch("v_depth2_diff30", VLEN)
                with self.tag_scope("preamble:group_diff"):
                    self.emit_vbroadcasts(
                        [
                            (v_depth2_diff10, diff2_10_s),
                            (v_depth2_diff20, diff2_20_s),
                            (v_depth2_diff30, diff2_30_s),
                        ]
                    )
            if diff3_10_s is not None:
                v_depth3_diff10 = self.alloc_scratch("v_depth3_diff10", VLEN)
                v_depth3_diff20 = self.alloc_scratch("v_depth3_diff20", VLEN)
                v_depth3_diff30 = self.alloc_scratch("v_depth3_diff30", VLEN)
                v_depth3_diff40 = self.alloc_scratch("v_depth3_diff40", VLEN)
                v_depth3_diff50 = self.alloc_scratch("v_depth3_diff50", VLEN)
                v_depth3_diff60 = self.alloc_scratch("v_depth3_diff60", VLEN)
                v_depth3_diff70 = self.alloc_scratch("v_depth3_diff70", VLEN)
                with self.tag_scope("preamble:group_diff"):
                    self.emit_vbroadcasts(
                        [
                            (v_depth3_diff10, diff3_10_s),
                            (v_depth3_diff20, diff3_20_s),
                            (v_depth3_diff30, diff3_30_s),
                            (v_depth3_diff40, diff3_40_s),
                            (v_depth3_diff50, diff3_50_s),
                            (v_depth3_diff60, diff3_60_s),
                            (v_depth3_diff70, diff3_70_s),
                        ]
                    )

        # Precompute per-depth base pointers for offset indexing on non-grouped depths.
        base_ptrs = [None] * (forest_height + 1)
        base_ptr_ops = []
        for depth in range(forest_height + 1):
            if use_shallow_grouping and depth <= shallow_group_depth:
                continue
            base_idx_const = self.scratch_const((1 << depth) - 1)
            base_ptr = self.alloc_scratch(f"base_ptr_d{depth}")
            base_ptr_ops.append(
                ("alu", ("+", base_ptr, self.scratch["forest_values_p"], base_idx_const))
            )
            base_ptrs[depth] = base_ptr
        with self.tag_scope("preamble:base_ptr"):
            for i in range(0, len(base_ptr_ops), SLOT_LIMITS["alu"]):
                self.emit_cycle(base_ptr_ops[i : i + SLOT_LIMITS["alu"]])

        idx_addr = self.alloc_scratch("idx_addr")
        val_addr = self.alloc_scratch("val_addr")
        idx_addr_b = [self.alloc_scratch(f"idx_addr_b_{i}") for i in range(blocks)]
        val_addr_b = [self.alloc_scratch(f"val_addr_b_{i}") for i in range(blocks)]

        idx_all = self.alloc_scratch("idx_all", blocks * VLEN)
        val_all = self.alloc_scratch("val_all", blocks * VLEN)

        if partition_grouping:
            batch_const = self.scratch_const(batch_size)

            pos_shift_const = self.scratch_const(16)
            pos_mask = self.scratch_const(0x00FF0000)
            idx_mask = self.scratch_const(0xFFFF)
            pos_mask8 = self.scratch_const(0xFF)

            vlen_const = self.scratch_const(VLEN)
            vlen_minus_one = self.scratch_const(VLEN - 1)

            group_val_vec = self.alloc_scratch("group_val_vec", VLEN)
            group_tmp1_vec = self.alloc_scratch("group_tmp1_vec", VLEN)
            group_tmp2_vec = self.alloc_scratch("group_tmp2_vec", VLEN)
            group_node_vec = self.alloc_scratch("group_node_vec", VLEN)

            # Pack positions into inp_indices (pos in bits 16-23)
            pos_i = self.alloc_scratch("pos_i")
            pos_addr = self.alloc_scratch("pos_addr")
            packed = self.alloc_scratch("packed")
            pos_shifted = self.alloc_scratch("pos_shifted")
            self.emit_cycle([("load", ("const", pos_i, 0))])
            self.label("pack_loop_start")
            self.emit_cycle([("alu", ("<", tmp1, pos_i, batch_const))])
            self.emit_cycle([("alu", ("==", tmp2, tmp1, zero_const))])
            self.add_cjump_rel(tmp2, "pack_loop_end")
            self.emit_cycle([("alu", ("+", pos_addr, self.scratch["inp_indices_p"], pos_i))])
            self.emit_cycle([("load", ("load", packed, pos_addr))])
            self.emit_cycle([("alu", ("<<", pos_shifted, pos_i, pos_shift_const))])
            self.emit_cycle([("alu", ("+", packed, packed, pos_shifted))])
            self.emit_cycle([("store", ("store", pos_addr, packed))])
            self.emit_cycle([("alu", ("+", pos_i, pos_i, one_const))])
            self.add_jump("pack_loop_start")
            self.label("pack_loop_end")

            # Pause to match first yield
            self.add("flow", ("pause",))

            counts_size = 1 << (group_depth + 1)
            counts_cur = self.alloc_scratch("counts_cur", counts_size)
            counts_next = self.alloc_scratch("counts_next", counts_size)

            self.emit_cycle([("alu", ("+", counts_cur, batch_const, zero_const))])
            for ci in range(1, counts_size):
                self.emit_cycle([("alu", ("+", counts_cur + ci, zero_const, zero_const))])
            for ci in range(counts_size):
                self.emit_cycle([("alu", ("+", counts_next + ci, zero_const, zero_const))])

            count = self.alloc_scratch("count")
            i = self.alloc_scratch("i")
            cur_off = self.alloc_scratch("cur_off")
            base_vals = self.alloc_scratch("base_vals")
            base_idx = self.alloc_scratch("base_idx")
            left = self.alloc_scratch("left")
            right = self.alloc_scratch("right")
            left_count = self.alloc_scratch("left_count")
            parity = self.alloc_scratch("parity")
            cond = self.alloc_scratch("cond")
            addr_l = self.alloc_scratch("addr_l")
            addr_r = self.alloc_scratch("addr_r")
            val_l = self.alloc_scratch("val_l")
            val_r = self.alloc_scratch("val_r")
            packed_l = self.alloc_scratch("packed_l")
            packed_r = self.alloc_scratch("packed_r")
            pos_bits = self.alloc_scratch("pos_bits")
            count_minus = self.alloc_scratch("count_minus")

            for depth in range(group_depth + 1):
                num_nodes = 1 << depth
                base = (1 << depth) - 1
                self.emit_cycle([("load", ("const", cur_off, 0))])
                for node_k in range(num_nodes):
                    node_idx = base + node_k
                    node_idx_const = self.scratch_const(node_idx)
                    left_idx_const = self.scratch_const(node_idx * 2 + 1)
                    right_idx_const = self.scratch_const(node_idx * 2 + 2)

                    self.emit_cycle([("alu", ("+", count, counts_cur + node_k, zero_const))])
                    self.emit_cycle([("alu", ("==", cond, count, zero_const))])
                    self.add_cjump_rel(cond, f"grp_d{depth}_n{node_k}_zero")

                    # base pointers for this group
                    self.emit_cycle([("alu", ("+", base_vals, self.scratch["inp_values_p"], cur_off))])
                    self.emit_cycle([("alu", ("+", base_idx, self.scratch["inp_indices_p"], cur_off))])

                    # node_val and broadcast
                    self.emit_cycle([("alu", ("+", addr_l, self.scratch["forest_values_p"], node_idx_const))])
                    self.emit_cycle([("load", ("load", tmp3, addr_l))])
                    self.emit_cycle([("valu", ("vbroadcast", group_node_vec, tmp3))])

                    # vector hash loop
                    self.emit_cycle([("load", ("const", i, 0))])
                    self.emit_cycle([("alu", ("<", cond, count, vlen_const))])
                    self.add_cjump_rel(cond, f"grp_d{depth}_n{node_k}_vec_end")
                    self.emit_cycle([("alu", ("-", count_minus, count, vlen_minus_one))])
                    self.label(f"grp_d{depth}_n{node_k}_vec")
                    self.emit_cycle([("alu", ("<", cond, i, count_minus))])
                    self.emit_cycle([("alu", ("==", tmp2, cond, zero_const))])
                    self.add_cjump_rel(tmp2, f"grp_d{depth}_n{node_k}_vec_end")
                    self.emit_cycle([("alu", ("+", addr_l, base_vals, i))])
                    self.emit_cycle([("load", ("vload", group_val_vec, addr_l))])
                    self.emit_cycle([("valu", ("^", group_val_vec, group_val_vec, group_node_vec))])
                    for entry in hash_plan:
                        if entry[0] == "fused":
                            _, v_mult, v_c1 = entry
                            self.emit_cycle(
                                [
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            group_val_vec,
                                            group_val_vec,
                                            v_mult,
                                            v_c1,
                                        ),
                                    )
                                ]
                            )
                        else:
                            _, op1, v_c1, op2, op3, v_c3 = entry
                            self.emit_cycle(
                                [("valu", (op1, group_tmp1_vec, group_val_vec, v_c1))]
                            )
                            self.emit_cycle(
                                [("valu", (op3, group_tmp2_vec, group_val_vec, v_c3))]
                            )
                            self.emit_cycle(
                                [
                                    (
                                        "valu",
                                        (op2, group_val_vec, group_tmp1_vec, group_tmp2_vec),
                                    )
                                ]
                            )
                    self.emit_cycle([("alu", ("+", addr_l, base_vals, i))])
                    self.emit_cycle([("store", ("vstore", addr_l, group_val_vec))])
                    self.emit_cycle([("alu", ("+", i, i, vlen_const))])
                    self.add_jump(f"grp_d{depth}_n{node_k}_vec")
                    self.label(f"grp_d{depth}_n{node_k}_vec_end")

                    # scalar tail
                    self.label(f"grp_d{depth}_n{node_k}_tail")
                    self.emit_cycle([("alu", ("<", cond, i, count))])
                    self.emit_cycle([("alu", ("==", tmp2, cond, zero_const))])
                    self.add_cjump_rel(tmp2, f"grp_d{depth}_n{node_k}_tail_end")
                    self.emit_cycle([("alu", ("+", addr_l, base_vals, i))])
                    self.emit_cycle([("load", ("load", val_l, addr_l))])
                    self.emit_cycle([("alu", ("^", val_l, val_l, tmp3))])
                    for op1, c1, op2, op3, c3 in hash_scalar_consts:
                        self.emit_cycle([("alu", (op1, tmp1, val_l, c1))])
                        self.emit_cycle([("alu", (op3, tmp2, val_l, c3))])
                        self.emit_cycle([("alu", (op2, val_l, tmp1, tmp2))])
                    self.emit_cycle([("store", ("store", addr_l, val_l))])
                    self.emit_cycle([("alu", ("+", i, i, one_const))])
                    self.add_jump(f"grp_d{depth}_n{node_k}_tail")
                    self.label(f"grp_d{depth}_n{node_k}_tail_end")

                    # in-place partition by parity
                    self.emit_cycle([("alu", ("+", left, cur_off, zero_const))])
                    self.emit_cycle([("alu", ("-", right, count, one_const))])
                    self.emit_cycle([("alu", ("+", right, right, cur_off))])
                    self.emit_cycle([("load", ("const", left_count, 0))])
                    self.label(f"grp_d{depth}_n{node_k}_part")
                    self.emit_cycle([("alu", ("+", tmp1, right, one_const))])
                    self.emit_cycle([("alu", ("<", cond, left, tmp1))])
                    self.emit_cycle([("alu", ("==", tmp2, cond, zero_const))])
                    self.add_cjump_rel(tmp2, f"grp_d{depth}_n{node_k}_part_end")

                    # load left value
                    self.emit_cycle([("alu", ("+", addr_l, self.scratch["inp_values_p"], left))])
                    self.emit_cycle([("load", ("load", val_l, addr_l))])
                    self.emit_cycle([("alu", ("&", parity, val_l, one_const))])
                    self.emit_cycle([("alu", ("==", cond, parity, zero_const))])
                    self.add_cjump_rel(cond, f"grp_d{depth}_n{node_k}_left_ok")

                    # parity == 1, check right
                    self.emit_cycle([("alu", ("+", addr_r, self.scratch["inp_values_p"], right))])
                    self.emit_cycle([("load", ("load", val_r, addr_r))])
                    self.emit_cycle([("alu", ("&", tmp1, val_r, one_const))])
                    self.emit_cycle([("alu", ("==", cond, tmp1, one_const))])
                    self.add_cjump_rel(cond, f"grp_d{depth}_n{node_k}_right_ok")

                    # swap left/right
                    self.emit_cycle([("alu", ("+", idx_addr, self.scratch["inp_indices_p"], left))])
                    self.emit_cycle([("load", ("load", packed_l, idx_addr))])
                    self.emit_cycle([("alu", ("+", idx_addr, self.scratch["inp_indices_p"], right))])
                    self.emit_cycle([("load", ("load", packed_r, idx_addr))])
                    self.emit_cycle([("store", ("store", addr_l, val_r))])
                    self.emit_cycle([("store", ("store", addr_r, val_l))])
                    self.emit_cycle([("alu", ("&", pos_bits, packed_r, pos_mask))])
                    self.emit_cycle([("alu", ("+", packed, pos_bits, left_idx_const))])
                    self.emit_cycle([("alu", ("+", idx_addr, self.scratch["inp_indices_p"], left))])
                    self.emit_cycle([("store", ("store", idx_addr, packed))])
                    self.emit_cycle([("alu", ("&", pos_bits, packed_l, pos_mask))])
                    self.emit_cycle([("alu", ("+", packed, pos_bits, right_idx_const))])
                    self.emit_cycle([("alu", ("+", idx_addr, self.scratch["inp_indices_p"], right))])
                    self.emit_cycle([("store", ("store", idx_addr, packed))])
                    self.emit_cycle([("alu", ("+", left, left, one_const))])
                    self.emit_cycle([("alu", ("-", right, right, one_const))])
                    self.emit_cycle([("alu", ("+", left_count, left_count, one_const))])
                    self.add_jump(f"grp_d{depth}_n{node_k}_part")

                    self.label(f"grp_d{depth}_n{node_k}_left_ok")
                    self.emit_cycle([("alu", ("+", idx_addr, self.scratch["inp_indices_p"], left))])
                    self.emit_cycle([("load", ("load", packed_l, idx_addr))])
                    self.emit_cycle([("alu", ("&", pos_bits, packed_l, pos_mask))])
                    self.emit_cycle([("alu", ("+", packed, pos_bits, left_idx_const))])
                    self.emit_cycle([("store", ("store", idx_addr, packed))])
                    self.emit_cycle([("alu", ("+", left, left, one_const))])
                    self.emit_cycle([("alu", ("+", left_count, left_count, one_const))])
                    self.add_jump(f"grp_d{depth}_n{node_k}_part")

                    self.label(f"grp_d{depth}_n{node_k}_right_ok")
                    self.emit_cycle([("alu", ("+", idx_addr, self.scratch["inp_indices_p"], right))])
                    self.emit_cycle([("load", ("load", packed_r, idx_addr))])
                    self.emit_cycle([("alu", ("&", pos_bits, packed_r, pos_mask))])
                    self.emit_cycle([("alu", ("+", packed, pos_bits, right_idx_const))])
                    self.emit_cycle([("store", ("store", idx_addr, packed))])
                    self.emit_cycle([("alu", ("-", right, right, one_const))])
                    self.add_jump(f"grp_d{depth}_n{node_k}_part")

                    self.label(f"grp_d{depth}_n{node_k}_part_end")
                    self.emit_cycle([("alu", ("+", counts_next + node_k * 2, left_count, zero_const))])
                    self.emit_cycle([("alu", ("-", tmp1, count, left_count))])
                    self.emit_cycle([("alu", ("+", counts_next + node_k * 2 + 1, tmp1, zero_const))])
                    self.emit_cycle([("alu", ("+", cur_off, cur_off, count))])
                    self.add_jump(f"grp_d{depth}_n{node_k}_end")

                    self.label(f"grp_d{depth}_n{node_k}_zero")
                    self.emit_cycle([("alu", ("+", counts_next + node_k * 2, zero_const, zero_const))])
                    self.emit_cycle([("alu", ("+", counts_next + node_k * 2 + 1, zero_const, zero_const))])
                    self.label(f"grp_d{depth}_n{node_k}_end")

                # prepare counts for next depth
                next_nodes = 1 << (depth + 1)
                for ci in range(next_nodes):
                    self.emit_cycle([("alu", ("+", counts_cur + ci, counts_next + ci, zero_const))])

            # Permute back to original order and unpack idx
            perm_i = self.alloc_scratch("perm_i")
            perm_pos = self.alloc_scratch("perm_pos")
            self.emit_cycle([("load", ("const", perm_i, 0))])
            self.label("perm_loop_start")
            self.emit_cycle([("alu", ("<", cond, perm_i, batch_const))])
            self.emit_cycle([("alu", ("==", tmp2, cond, zero_const))])
            self.add_cjump_rel(tmp2, "perm_loop_end")

            self.emit_cycle([("alu", ("+", idx_addr, self.scratch["inp_indices_p"], perm_i))])
            self.emit_cycle([("load", ("load", packed_l, idx_addr))])
            self.emit_cycle([("alu", (">>", tmp1, packed_l, pos_shift_const))])
            self.emit_cycle([("alu", ("&", perm_pos, tmp1, pos_mask8))])
            self.emit_cycle([("alu", ("==", cond, perm_pos, perm_i))])
            self.add_cjump_rel(cond, "perm_fix")

            self.label("perm_swap")
            self.emit_cycle([("alu", ("+", idx_addr, self.scratch["inp_indices_p"], perm_pos))])
            self.emit_cycle([("load", ("load", packed_r, idx_addr))])
            self.emit_cycle([("alu", ("+", addr_l, self.scratch["inp_values_p"], perm_i))])
            self.emit_cycle([("load", ("load", val_l, addr_l))])
            self.emit_cycle([("alu", ("+", addr_r, self.scratch["inp_values_p"], perm_pos))])
            self.emit_cycle([("load", ("load", val_r, addr_r))])
            self.emit_cycle([("store", ("store", addr_l, val_r))])
            self.emit_cycle([("store", ("store", addr_r, val_l))])
            self.emit_cycle([("alu", ("+", idx_addr, self.scratch["inp_indices_p"], perm_i))])
            self.emit_cycle([("store", ("store", idx_addr, packed_r))])
            self.emit_cycle([("alu", ("+", idx_addr, self.scratch["inp_indices_p"], perm_pos))])
            self.emit_cycle([("store", ("store", idx_addr, packed_l))])
            self.emit_cycle([("alu", ("+", packed_l, packed_r, zero_const))])
            self.emit_cycle([("alu", (">>", tmp1, packed_l, pos_shift_const))])
            self.emit_cycle([("alu", ("&", perm_pos, tmp1, pos_mask8))])
            self.emit_cycle([("alu", ("==", cond, perm_pos, perm_i))])
            self.add_cjump_rel(cond, "perm_fix")
            self.add_jump("perm_swap")

            self.label("perm_fix")
            self.emit_cycle([("alu", ("&", tmp1, packed_l, idx_mask))])
            self.emit_cycle([("alu", ("+", idx_addr, self.scratch["inp_indices_p"], perm_i))])
            self.emit_cycle([("store", ("store", idx_addr, tmp1))])
            self.emit_cycle([("alu", ("+", perm_i, perm_i, one_const))])
            self.add_jump("perm_loop_start")
            self.label("perm_loop_end")

            # Load idx/val into scratch directly after regrouping
            ops_by_block = []
            for bi in range(blocks):
                off_const = block_offsets[bi]
                ops = [
                    ("alu", ("+", idx_addr_b[bi], self.scratch["inp_indices_p"], off_const)),
                    ("alu", ("+", val_addr_b[bi], self.scratch["inp_values_p"], off_const)),
                    ("load", ("vload", idx_all + bi * VLEN, idx_addr_b[bi])),
                    ("load", ("vload", val_all + bi * VLEN, val_addr_b[bi])),
                ]
                ops_by_block.append(ops)
            self.schedule_wave(ops_by_block, rr_start=0)

        else:
            # Initial vloads are scheduled together with rounds below.
            pass

        start_round = group_depth + 1 if partition_grouping else 0
        start_depth = start_round % (forest_height + 1)
        start_base_idx_const = None
        if start_depth:
            start_base_idx_const = self.scratch_const((1 << start_depth) - 1)

        interleave_addr_load = os.getenv("INTERLEAVE_ADDR_LOAD", "0") == "1"
        trim_last_wave = os.getenv("TRIM_LAST_WAVE", "0") == "1"
        wave_size = int(os.getenv("WAVE_SIZE", "468"))
        if flow_select_depth3 or valu_tree_depth3:
            num_tmp = 4
        elif flow_select_depth2:
            num_tmp = 3
        else:
            num_tmp = 2
        max_wave = (SCRATCH_SIZE - self.scratch_ptr) // (VLEN * num_tmp)
        if max_wave < 1:
            raise ValueError("Not enough scratch for wave temp vectors")
        scratch_wave = min(blocks, wave_size)
        if scratch_wave > max_wave:
            wave_size = max_wave
            scratch_wave = wave_size
        waves = (blocks + wave_size - 1) // wave_size

        # Per-block scratch for a wave
        tmp1_vec_b = [self.alloc_scratch(f"tmp1_vec_{i}", VLEN) for i in range(scratch_wave)]
        tmp2_vec_b = [self.alloc_scratch(f"tmp2_vec_{i}", VLEN) for i in range(scratch_wave)]
        if flow_select_depth2 or flow_select_depth3 or valu_tree_depth3:
            tmp3_vec_b = [
                self.alloc_scratch(f"tmp3_vec_{i}", VLEN) for i in range(scratch_wave)
            ]
        if flow_select_depth3 or valu_tree_depth3:
            tmp4_vec_b = [
                self.alloc_scratch(f"tmp4_vec_{i}", VLEN) for i in range(scratch_wave)
            ]

        round_plans = []
        for round in range(start_round, rounds):
            depth = round % (forest_height + 1)
            use_grouping_round = use_shallow_grouping and depth <= shallow_group_depth
            if use_grouping_round:
                num_nodes = 1 << depth
                depth_offset = depth_offsets[depth]
            else:
                num_nodes = 0
                depth_offset = 0
            round_plans.append((depth, use_grouping_round, num_nodes, depth_offset))

        if not partition_grouping:
            # Pause instructions are matched up with yield statements in the reference
            # kernel to let you debug at intermediate steps. The testing harness in this
            # file requires these match up to the reference kernel's yields, but the
            # submission harness ignores them.
            self.add("flow", ("pause",))

        for wi in range(waves):
            ops_by_block = []
            base_block = wi * wave_size
            wave_blocks = min(wave_size, blocks - base_block) if trim_last_wave else wave_size
            for bi in range(wave_blocks):
                block_idx = base_block + bi
                if block_idx >= blocks:
                    ops_by_block.append([])
                    continue
                idx_vec = idx_all + block_idx * VLEN
                val_vec = val_all + block_idx * VLEN
                ops = []
                def add_op(engine, slot, tag):
                    ops.append((engine, slot, tag))
                if not partition_grouping:
                    off_const = block_offsets[block_idx]
                    add_op(
                        "alu",
                        (
                            "+",
                            idx_addr_b[block_idx],
                            self.scratch["inp_indices_p"],
                            off_const,
                        ),
                        "init:addr",
                    )
                    add_op(
                        "alu",
                        (
                            "+",
                            val_addr_b[block_idx],
                            self.scratch["inp_values_p"],
                            off_const,
                        ),
                        "init:addr",
                    )
                    add_op("load", ("vload", idx_vec, idx_addr_b[block_idx]), "init:vload")
                    add_op("load", ("vload", val_vec, val_addr_b[block_idx]), "init:vload")
                if start_base_idx_const is not None:
                    for lane in range(VLEN):
                        add_op(
                            "alu",
                            (
                                "-",
                                idx_vec + lane,
                                idx_vec + lane,
                                start_base_idx_const,
                            ),
                            "idx:base_off",
                        )
                for (depth, use_grouping_round, num_nodes, depth_offset) in round_plans:
                    node_val_src = None
                    if use_grouping_round:
                        if depth == 0:
                            node_val_src = v_node_val_all + depth_offset
                        elif depth == 1:
                            if flow_select_depth1:
                                # off is 0/1; use it directly as mask
                                add_op(
                                    "flow",
                                    (
                                        "vselect",
                                        tmp2_vec_b[bi],
                                        idx_vec,
                                        v_node_val_all + depth_offset + VLEN,
                                        v_node_val_all + depth_offset,
                                    ),
                                    "group:d1",
                                )
                            else:
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        tmp2_vec_b[bi],
                                        idx_vec,
                                        v_depth1_diff,
                                        v_node_val_all + depth_offset,
                                    ),
                                    "group:d1",
                                )
                            node_val_src = tmp2_vec_b[bi]
                        else:
                            if flow_select_depth2 and depth == 2:
                                # b0 = off & 1, b1 = off >> 1
                                add_op(
                                    "valu",
                                    ("&", tmp1_vec_b[bi], idx_vec, v_one),
                                    "group:d2",
                                )
                                add_op(
                                    "valu",
                                    (">>", tmp2_vec_b[bi], idx_vec, v_one),
                                    "group:d2",
                                )
                                # left = select(node1, node0) by b0
                                add_op(
                                    "flow",
                                    (
                                        "vselect",
                                        tmp3_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        v_node_val_all + depth_offset + VLEN,
                                        v_node_val_all + depth_offset,
                                    ),
                                    "group:d2",
                                )
                                # right = select(node3, node2) by b0
                                add_op(
                                    "flow",
                                    (
                                        "vselect",
                                        tmp1_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        v_node_val_all + depth_offset + 3 * VLEN,
                                        v_node_val_all + depth_offset + 2 * VLEN,
                                    ),
                                    "group:d2",
                                )
                                # node_val = select(right, left) by b1
                                add_op(
                                    "flow",
                                    (
                                        "vselect",
                                        tmp2_vec_b[bi],
                                        tmp2_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        tmp3_vec_b[bi],
                                    ),
                                    "group:d2",
                                )
                                node_val_src = tmp2_vec_b[bi]
                            elif (
                                depth == 2
                                and fast_depth2_select
                                and v_depth2_diff10 is not None
                            ):
                                # Start from node0, then add diffs for the other nodes.
                                add_op(
                                    "valu",
                                    ("==", tmp1_vec_b[bi], idx_vec, v_one),
                                    "group:d2",
                                )
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        tmp2_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        v_depth2_diff10,
                                        v_node_val_all + depth_offset,
                                    ),
                                    "group:d2",
                                )
                                add_op(
                                    "valu",
                                    ("==", tmp1_vec_b[bi], idx_vec, v_two),
                                    "group:d2",
                                )
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        tmp2_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        v_depth2_diff20,
                                        tmp2_vec_b[bi],
                                    ),
                                    "group:d2",
                                )
                                add_op(
                                    "valu",
                                    ("==", tmp1_vec_b[bi], idx_vec, v_three),
                                    "group:d2",
                                )
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        tmp2_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        v_depth2_diff30,
                                        tmp2_vec_b[bi],
                                    ),
                                    "group:d2",
                                )
                                node_val_src = tmp2_vec_b[bi]
                            elif depth == 3 and valu_tree_depth3:
                                # Tree-based VALU selection using bit masks.
                                # b0 = idx & 1
                                add_op(
                                    "valu",
                                    ("&", tmp1_vec_b[bi], idx_vec, v_one),
                                    "group:d3",
                                )
                                # t0 = n0 + b0*(n1 - n0)
                                add_op(
                                    "valu",
                                    (
                                        "-",
                                        tmp2_vec_b[bi],
                                        v_node_val_all + depth_offset + VLEN,
                                        v_node_val_all + depth_offset,
                                    ),
                                    "group:d3",
                                )
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        tmp2_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        tmp2_vec_b[bi],
                                        v_node_val_all + depth_offset,
                                    ),
                                    "group:d3",
                                )
                                # t1 = n2 + b0*(n3 - n2)
                                add_op(
                                    "valu",
                                    (
                                        "-",
                                        tmp3_vec_b[bi],
                                        v_node_val_all + depth_offset + 3 * VLEN,
                                        v_node_val_all + depth_offset + 2 * VLEN,
                                    ),
                                    "group:d3",
                                )
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        tmp3_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        tmp3_vec_b[bi],
                                        v_node_val_all + depth_offset + 2 * VLEN,
                                    ),
                                    "group:d3",
                                )
                                # b1 = (idx & 2) >> 1
                                add_op(
                                    "valu",
                                    ("&", tmp4_vec_b[bi], idx_vec, v_two),
                                    "group:d3",
                                )
                                add_op(
                                    "valu",
                                    (">>", tmp4_vec_b[bi], tmp4_vec_b[bi], v_one),
                                    "group:d3",
                                )
                                # u0 = t0 + b1*(t1 - t0)
                                add_op(
                                    "valu",
                                    ("-", tmp3_vec_b[bi], tmp3_vec_b[bi], tmp2_vec_b[bi]),
                                    "group:d3",
                                )
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        tmp2_vec_b[bi],
                                        tmp4_vec_b[bi],
                                        tmp3_vec_b[bi],
                                        tmp2_vec_b[bi],
                                    ),
                                    "group:d3",
                                )
                                # t2 = n4 + b0*(n5 - n4)
                                add_op(
                                    "valu",
                                    (
                                        "-",
                                        tmp3_vec_b[bi],
                                        v_node_val_all + depth_offset + 5 * VLEN,
                                        v_node_val_all + depth_offset + 4 * VLEN,
                                    ),
                                    "group:d3",
                                )
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        tmp3_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        tmp3_vec_b[bi],
                                        v_node_val_all + depth_offset + 4 * VLEN,
                                    ),
                                    "group:d3",
                                )
                                # t3 = n6 + b0*(n7 - n6) (overwrite b0 in tmp1 after this)
                                add_op(
                                    "valu",
                                    (
                                        "-",
                                        tmp4_vec_b[bi],
                                        v_node_val_all + depth_offset + 7 * VLEN,
                                        v_node_val_all + depth_offset + 6 * VLEN,
                                    ),
                                    "group:d3",
                                )
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        tmp4_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        tmp4_vec_b[bi],
                                        v_node_val_all + depth_offset + 6 * VLEN,
                                    ),
                                    "group:d3",
                                )
                                # b1 = (idx & 2) >> 1 (recompute into tmp1)
                                add_op(
                                    "valu",
                                    ("&", tmp1_vec_b[bi], idx_vec, v_two),
                                    "group:d3",
                                )
                                add_op(
                                    "valu",
                                    (">>", tmp1_vec_b[bi], tmp1_vec_b[bi], v_one),
                                    "group:d3",
                                )
                                # u1 = t2 + b1*(t3 - t2)
                                add_op(
                                    "valu",
                                    ("-", tmp4_vec_b[bi], tmp4_vec_b[bi], tmp3_vec_b[bi]),
                                    "group:d3",
                                )
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        tmp3_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        tmp4_vec_b[bi],
                                        tmp3_vec_b[bi],
                                    ),
                                    "group:d3",
                                )
                                # b2 = (idx & 4) >> 2
                                add_op(
                                    "valu",
                                    ("&", tmp1_vec_b[bi], idx_vec, v_four),
                                    "group:d3",
                                )
                                add_op(
                                    "valu",
                                    (">>", tmp1_vec_b[bi], tmp1_vec_b[bi], v_two),
                                    "group:d3",
                                )
                                # out = u0 + b2*(u1 - u0)
                                add_op(
                                    "valu",
                                    ("-", tmp3_vec_b[bi], tmp3_vec_b[bi], tmp2_vec_b[bi]),
                                    "group:d3",
                                )
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        tmp2_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        tmp3_vec_b[bi],
                                        tmp2_vec_b[bi],
                                    ),
                                    "group:d3",
                                )
                                node_val_src = tmp2_vec_b[bi]
                            elif depth == 3 and flow_select_depth3:
                                # b0 = off & 1
                                add_op(
                                    "valu",
                                    ("&", tmp1_vec_b[bi], idx_vec, v_one),
                                    "group:d3",
                                )
                                # s0 = select(node1, node0) by b0
                                add_op(
                                    "flow",
                                    (
                                        "vselect",
                                        tmp2_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        v_node_val_all + depth_offset + VLEN,
                                        v_node_val_all + depth_offset,
                                    ),
                                    "group:d3",
                                )
                                # s1 = select(node3, node2) by b0
                                add_op(
                                    "flow",
                                    (
                                        "vselect",
                                        tmp3_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        v_node_val_all + depth_offset + 3 * VLEN,
                                        v_node_val_all + depth_offset + 2 * VLEN,
                                    ),
                                    "group:d3",
                                )
                                # b1 = off & 2
                                add_op(
                                    "valu",
                                    ("&", tmp4_vec_b[bi], idx_vec, v_two),
                                    "group:d3",
                                )
                                # t0 = select(s1, s0) by b1
                                add_op(
                                    "flow",
                                    (
                                        "vselect",
                                        tmp2_vec_b[bi],
                                        tmp4_vec_b[bi],
                                        tmp3_vec_b[bi],
                                        tmp2_vec_b[bi],
                                    ),
                                    "group:d3",
                                )
                                # s2 = select(node5, node4) by b0
                                add_op(
                                    "flow",
                                    (
                                        "vselect",
                                        tmp3_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        v_node_val_all + depth_offset + 5 * VLEN,
                                        v_node_val_all + depth_offset + 4 * VLEN,
                                    ),
                                    "group:d3",
                                )
                                # s3 = select(node7, node6) by b0 (dest overwrites b0)
                                add_op(
                                    "flow",
                                    (
                                        "vselect",
                                        tmp1_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        v_node_val_all + depth_offset + 7 * VLEN,
                                        v_node_val_all + depth_offset + 6 * VLEN,
                                    ),
                                    "group:d3",
                                )
                                # t1 = select(s3, s2) by b1
                                add_op(
                                    "flow",
                                    (
                                        "vselect",
                                        tmp3_vec_b[bi],
                                        tmp4_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        tmp3_vec_b[bi],
                                    ),
                                    "group:d3",
                                )
                                # b2 = off & 4
                                add_op(
                                    "valu",
                                    ("&", tmp4_vec_b[bi], idx_vec, v_four),
                                    "group:d3",
                                )
                                # node_val = select(t1, t0) by b2
                                add_op(
                                    "flow",
                                    (
                                        "vselect",
                                        tmp2_vec_b[bi],
                                        tmp4_vec_b[bi],
                                        tmp3_vec_b[bi],
                                        tmp2_vec_b[bi],
                                    ),
                                    "group:d3",
                                )
                                node_val_src = tmp2_vec_b[bi]
                            elif (
                                depth == 3
                                and fast_depth3_select
                                and v_depth3_diff10 is not None
                            ):
                                # Start from node0, then add diffs for the other nodes.
                                add_op(
                                    "valu",
                                    ("==", tmp1_vec_b[bi], idx_vec, v_one),
                                    "group:d3",
                                )
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        tmp2_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        v_depth3_diff10,
                                        v_node_val_all + depth_offset,
                                    ),
                                    "group:d3",
                                )
                                add_op(
                                    "valu",
                                    ("==", tmp1_vec_b[bi], idx_vec, v_two),
                                    "group:d3",
                                )
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        tmp2_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        v_depth3_diff20,
                                        tmp2_vec_b[bi],
                                    ),
                                    "group:d3",
                                )
                                add_op(
                                    "valu",
                                    ("==", tmp1_vec_b[bi], idx_vec, v_three),
                                    "group:d3",
                                )
                                add_op(
                                    "valu",
                                    (
                                        "multiply_add",
                                        tmp2_vec_b[bi],
                                        tmp1_vec_b[bi],
                                        v_depth3_diff30,
                                        tmp2_vec_b[bi],
                                    ),
                                    "group:d3",
                                )
                                for const_idx, diff_vec in (
                                    (4, v_depth3_diff40),
                                    (5, v_depth3_diff50),
                                    (6, v_depth3_diff60),
                                    (7, v_depth3_diff70),
                                ):
                                    add_op(
                                        "valu",
                                        (
                                            "vbroadcast",
                                            tmp1_vec_b[bi],
                                            node_idx_consts[const_idx],
                                        ),
                                        "group:d3",
                                    )
                                    add_op(
                                        "valu",
                                        ("==", tmp1_vec_b[bi], idx_vec, tmp1_vec_b[bi]),
                                        "group:d3",
                                    )
                                    add_op(
                                        "valu",
                                        (
                                            "multiply_add",
                                            tmp2_vec_b[bi],
                                            tmp1_vec_b[bi],
                                            diff_vec,
                                            tmp2_vec_b[bi],
                                        ),
                                        "group:d3",
                                    )
                                node_val_src = tmp2_vec_b[bi]
                            else:
                                # tmp2_vec_b holds node_val accumulator
                                add_op(
                                    "valu",
                                    ("+", tmp2_vec_b[bi], v_zero, v_zero),
                                    f"group:d{depth}",
                                )
                                for node_k in range(num_nodes):
                                    add_op(
                                        "valu",
                                        (
                                            "==",
                                            tmp1_vec_b[bi],
                                            idx_vec,
                                            v_node_off_all
                                            + depth_offset
                                            + node_k * VLEN,
                                        ),
                                        f"group:d{depth}",
                                    )
                                    add_op(
                                        "valu",
                                        (
                                            "multiply_add",
                                            tmp2_vec_b[bi],
                                            tmp1_vec_b[bi],
                                            v_node_val_all
                                            + depth_offset
                                            + node_k * VLEN,
                                            tmp2_vec_b[bi],
                                        ),
                                        f"group:d{depth}",
                                    )
                                node_val_src = tmp2_vec_b[bi]
                    else:
                        base_ptr = base_ptrs[depth]
                        if interleave_addr_load:
                            for lane in range(VLEN):
                                add_op(
                                    "alu",
                                    (
                                        "+",
                                        tmp1_vec_b[bi] + lane,
                                        idx_vec + lane,
                                        base_ptr,
                                    ),
                                    "gather:addr",
                                )
                                add_op(
                                    "load",
                                    ("load_offset", tmp2_vec_b[bi], tmp1_vec_b[bi], lane),
                                    "gather:load",
                                )
                        else:
                            for lane in range(VLEN):
                                add_op(
                                    "alu",
                                    (
                                        "+",
                                        tmp1_vec_b[bi] + lane,
                                        idx_vec + lane,
                                        base_ptr,
                                    ),
                                    "gather:addr",
                                )
                            for lane in range(VLEN):
                                add_op(
                                    "load",
                                    ("load_offset", tmp2_vec_b[bi], tmp1_vec_b[bi], lane),
                                    "gather:load",
                                )
                        node_val_src = tmp2_vec_b[bi]
                    add_op("valu", ("^", val_vec, val_vec, node_val_src), "xor")

                    for entry in hash_plan:
                        if entry[0] == "fused":
                            _, v_mult, v_c1 = entry
                            add_op(
                                "valu",
                                ("multiply_add", val_vec, val_vec, v_mult, v_c1),
                                "hash",
                            )
                        else:
                            _, op1, v_c1, op2, op3, v_c3 = entry
                            add_op("valu", (op1, tmp1_vec_b[bi], val_vec, v_c1), "hash")
                            add_op("valu", (op3, tmp2_vec_b[bi], val_vec, v_c3), "hash")
                            add_op(
                                "valu", (op2, val_vec, tmp1_vec_b[bi], tmp2_vec_b[bi]), "hash"
                            )

                    if depth == forest_height:
                        add_op("valu", ("+", idx_vec, v_zero, v_zero), "idx_reset")
                    else:
                        add_op("valu", ("&", tmp1_vec_b[bi], val_vec, v_one), "idx_update")
                        add_op(
                            "valu",
                            ("multiply_add", idx_vec, idx_vec, v_two, tmp1_vec_b[bi]),
                            "idx_update",
                        )

                if not partition_grouping:
                    add_op("store", ("vstore", idx_addr_b[block_idx], idx_vec), "store_idx")
                    add_op("store", ("vstore", val_addr_b[block_idx], val_vec), "store_val")

                ops_by_block.append(ops)

            self.schedule_wave(ops_by_block, rr_start=0)

        if partition_grouping:
            ops_by_block = []
            for bi in range(blocks):
                off_const = block_offsets[bi]
                ops = [
                    ("alu", ("+", idx_addr_b[bi], self.scratch["inp_indices_p"], off_const)),
                    ("alu", ("+", val_addr_b[bi], self.scratch["inp_values_p"], off_const)),
                ]
                ops.extend(
                    [
                        ("store", ("vstore", idx_addr_b[bi], idx_all + bi * VLEN)),
                        ("store", ("vstore", val_addr_b[bi], val_all + bi * VLEN)),
                    ]
                )
                ops_by_block.append(ops)
            self.schedule_wave(ops_by_block, rr_start=0)

        # Scalar tail for non-multiple of VLEN
        if tail_start < batch_size:
            tmp_idx = self.alloc_scratch("tmp_idx")
            tmp_val = self.alloc_scratch("tmp_val")
            tmp_node_val = self.alloc_scratch("tmp_node_val")
            tmp_addr = self.alloc_scratch("tmp_addr")
            for round in range(rounds):
                for i in range(tail_start, batch_size):
                    i_const = self.scratch_const(i)
                    # idx = mem[inp_indices_p + i]
                    self.emit_cycle(
                        [
                            (
                                "alu",
                                ("+", tmp_addr, self.scratch["inp_indices_p"], i_const),
                            )
                        ]
                    )
                    self.emit_cycle([("load", ("load", tmp_idx, tmp_addr))])
                    # val = mem[inp_values_p + i]
                    self.emit_cycle(
                        [
                            (
                                "alu",
                                ("+", tmp_addr, self.scratch["inp_values_p"], i_const),
                            )
                        ]
                    )
                    self.emit_cycle([("load", ("load", tmp_val, tmp_addr))])
                    # node_val = mem[forest_values_p + idx]
                    self.emit_cycle(
                        [
                            (
                                "alu",
                                ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx),
                            )
                        ]
                    )
                    self.emit_cycle([("load", ("load", tmp_node_val, tmp_addr))])
                    # val = myhash(val ^ node_val)
                    self.emit_cycle([("alu", ("^", tmp_val, tmp_val, tmp_node_val))])
                    for op1, val1, op2, op3, val3 in HASH_STAGES:
                        c1 = self.scratch_const(val1)
                        c3 = self.scratch_const(val3)
                        self.emit_cycle([("alu", (op1, tmp1, tmp_val, c1))])
                        self.emit_cycle([("alu", (op3, tmp2, tmp_val, c3))])
                        self.emit_cycle([("alu", (op2, tmp_val, tmp1, tmp2))])
                    # idx = 2*idx + (1 if val % 2 == 0 else 2)
                    self.emit_cycle([("alu", ("&", tmp1, tmp_val, one_const))])
                    self.emit_cycle([("alu", ("+", tmp3, tmp1, one_const))])
                    self.emit_cycle([("alu", ("*", tmp_idx, tmp_idx, two_const))])
                    self.emit_cycle([("alu", ("+", tmp_idx, tmp_idx, tmp3))])
                    # idx = 0 if idx >= n_nodes else idx
                    self.emit_cycle(
                        [("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"]))]
                    )
                    self.emit_cycle([("alu", ("*", tmp_idx, tmp_idx, tmp1))])
                    # mem[inp_indices_p + i] = idx
                    self.emit_cycle(
                        [
                            (
                                "alu",
                                ("+", tmp_addr, self.scratch["inp_indices_p"], i_const),
                            )
                        ]
                    )
                    self.emit_cycle([("store", ("store", tmp_addr, tmp_idx))])
                    # mem[inp_values_p + i] = val
                    self.emit_cycle(
                        [
                            (
                                "alu",
                                ("+", tmp_addr, self.scratch["inp_values_p"], i_const),
                            )
                        ]
                    )
                    self.emit_cycle([("store", ("store", tmp_addr, tmp_val))])

        # Required to match with the yield in reference_kernel2
        if self.instrs and "flow" not in self.instrs[-1]:
            self.instrs[-1].setdefault("flow", []).append(("pause",))
        else:
            self.instrs.append({"flow": [("pause",)]})
        self.resolve_fixups()


def compute_slot_stats(instrs: list[dict]):
    total_slots = defaultdict(int)
    max_bundle = defaultdict(int)
    total_bundles = 0
    total_cycles = 0

    for bundle in instrs:
        total_bundles += 1
        if any(engine != "debug" for engine in bundle.keys()):
            total_cycles += 1
        for engine, slots in bundle.items():
            total_slots[engine] += len(slots)
            if len(slots) > max_bundle[engine]:
                max_bundle[engine] = len(slots)

    lower_bounds = {
        engine: total_slots[engine] / SLOT_LIMITS[engine]
        for engine in total_slots
    }
    theoretical_lb = max(lower_bounds.values(), default=0.0)

    return {
        "bundles": total_bundles,
        "cycles": total_cycles,
        "total_slots": dict(total_slots),
        "max_bundle": dict(max_bundle),
        "lower_bounds": lower_bounds,
        "theoretical_lb": theoretical_lb,
    }


def print_slot_stats(stats: dict):
    print(f"Instruction bundles: {stats['bundles']}")
    print(f"Cycles (non-debug bundles): {stats['cycles']}")
    print("\nSlot counts by engine:")
    for engine in ["alu", "valu", "load", "store", "flow", "debug"]:
        slots = stats["total_slots"].get(engine, 0)
        if slots == 0:
            continue
        limit = SLOT_LIMITS[engine]
        util = slots / (max(stats["cycles"], 1) * limit)
        lower = stats["lower_bounds"].get(engine, 0.0)
        max_bundle = stats["max_bundle"].get(engine, 0)
        print(
            f"- {engine:5s}: slots={slots:6d} "
            f"max/bundle={max_bundle:2d} "
            f"util={util:6.2%} "
            f"lb_cycles={lower:8.1f}"
        )
    print(
        f"\nTheoretical lower bound (slot count / limit): {stats['theoretical_lb']:.1f} cycles"
    )

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
    stats: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    if stats:
        print_slot_stats(compute_slot_stats(kb.instrs))
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
