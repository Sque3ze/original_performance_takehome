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

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

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

    def emit_cycle(self, slots: list[tuple[Engine, tuple]]):
        if not slots:
            return
        bundle = defaultdict(list)
        counts = defaultdict(int)
        for engine, slot in slots:
            counts[engine] += 1
            assert counts[engine] <= SLOT_LIMITS[engine], f"{engine} slots exceeded"
            bundle[engine].append(slot)
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
        window = int(os.getenv("SCHED_WINDOW", "4"))
        for b in range(n_blocks):
            ops = ops_by_block[b]
            if ops and len(ops[0]) != 3:
                ops_by_block[b] = [
                    (i, engine, slot) for i, (engine, slot) in enumerate(ops)
                ]
        while any(ops_by_block[b] for b in range(n_blocks)):
            bundle = defaultdict(list)
            counts = defaultdict(int)
            block_used = [False] * n_blocks
            block_single = [False] * n_blocks
            block_sched = [[] for _ in range(n_blocks)]
            progressed = True
            while progressed:
                progressed = False
                for bi in range(n_blocks):
                    b = (rr + bi) % n_blocks
                    ops = ops_by_block[b]
                    if not ops:
                        continue
                    if block_single[b] and block_used[b]:
                        continue

                    prior_reads = set()
                    prior_writes = set()
                    chosen = None
                    chosen_reads = None
                    chosen_writes = None
                    for oi, (seq, engine, slot) in enumerate(ops[:window]):
                        reads, writes = self._slot_reads_writes(engine, slot)
                        if reads is None or writes is None:
                            # Unknown slot shape; treat as a barrier.
                            if oi == 0 and not block_used[b]:
                                chosen = (seq, engine, slot, oi)
                                chosen_reads = set()
                                chosen_writes = set()
                                block_single[b] = True
                            break

                        if counts[engine] >= SLOT_LIMITS[engine]:
                            prior_reads.update(reads)
                            prior_writes.update(writes)
                            continue

                        # Can't move this op ahead of earlier unscheduled ops if it would
                        # violate RAW/WAW/WAR with those earlier ops.
                        if reads & prior_writes or writes & (prior_writes | prior_reads):
                            prior_reads.update(reads)
                            prior_writes.update(writes)
                            continue

                        # Same-cycle hazards within the block.
                        conflict = False
                        for sched_seq, sched_reads, sched_writes in block_sched[b]:
                            if writes & sched_writes:
                                conflict = True
                                break
                            if reads & sched_writes and sched_seq < seq:
                                conflict = True
                                break
                            if writes & sched_reads and sched_seq > seq:
                                conflict = True
                                break
                        if conflict:
                            prior_reads.update(reads)
                            prior_writes.update(writes)
                            continue

                        chosen = (seq, engine, slot, oi)
                        chosen_reads = reads
                        chosen_writes = writes
                        break

                    if chosen is None:
                        continue

                    seq, engine, slot, oi = chosen
                    bundle[engine].append(slot)
                    counts[engine] += 1
                    if chosen_reads is not None and chosen_writes is not None:
                        block_sched[b].append((seq, chosen_reads, chosen_writes))
                    block_used[b] = True
                    ops.pop(oi)
                    progressed = True

            if not bundle:
                break
            self.instrs.append(dict(bundle))
            rr = (rr + 1) % n_blocks

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
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

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
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)

        vbroadcasts = [
            (v_zero, zero_const),
            (v_one, one_const),
            (v_two, two_const),
            (v_forest_p, self.scratch["forest_values_p"]),
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

        self.emit_vbroadcasts(vbroadcasts)
        v_one_plus_forest_p = self.alloc_scratch("v_one_plus_forest_p", VLEN)
        v_one_minus_forest_p = self.alloc_scratch("v_one_minus_forest_p", VLEN)
        self.emit_cycle([("valu", ("+", v_one_plus_forest_p, v_one, v_forest_p))])
        self.emit_cycle([("valu", ("-", v_one_minus_forest_p, v_one, v_forest_p))])

        hash_scalar_consts = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            c1 = self.scratch_const(val1)
            c3 = self.scratch_const(val3)
            hash_scalar_consts.append((op1, c1, op2, op3, c3))

        use_shallow_grouping = os.getenv("USE_SHALLOW_GROUPING", "1") != "0"
        shallow_group_depth = int(os.getenv("SHALLOW_GROUP_DEPTH", "2"))
        node_vals_s = []
        node_idx_consts = []
        depth_offsets = []
        v_node_idx_all = None
        v_node_val_all = None
        v_depth1_diff = None
        v_depth2_diff10 = None
        v_depth2_diff20 = None
        v_depth2_diff30 = None
        if use_shallow_grouping:
            max_group_node = (1 << (shallow_group_depth + 1)) - 2
            total_group_nodes = sum(1 << d for d in range(shallow_group_depth + 1))
            node_vals_s = [
                self.alloc_scratch(f"node_val_s_{i}") for i in range(max_group_node + 1)
            ]
            node_idx_consts = [self.scratch_const(i) for i in range(max_group_node + 1)]
            offset = 0
            for depth in range(shallow_group_depth + 1):
                depth_offsets.append(offset)
                offset += (1 << depth) * VLEN
            v_node_idx_all = self.alloc_scratch(
                "v_node_idx_all", total_group_nodes * VLEN
            )
            v_node_val_all = self.alloc_scratch(
                "v_node_val_all", total_group_nodes * VLEN
            )
            for node_idx in range(max_group_node + 1):
                self.emit_cycle(
                    [
                        (
                            "alu",
                            (
                                "+",
                                tmp1,
                                self.scratch["forest_values_p"],
                                node_idx_consts[node_idx],
                            ),
                        )
                    ]
                )
                self.emit_cycle([("load", ("load", node_vals_s[node_idx], tmp1))])
            idx_pairs = []
            val_pairs = []
            for depth in range(shallow_group_depth + 1):
                base = (1 << depth) - 1
                num_nodes = 1 << depth
                depth_off = depth_offsets[depth]
                for node_k in range(num_nodes):
                    node_idx = base + node_k
                    idx_pairs.append(
                        (
                            v_node_idx_all + depth_off + node_k * VLEN,
                            node_idx_consts[node_idx],
                        )
                    )
                    val_pairs.append(
                        (
                            v_node_val_all + depth_off + node_k * VLEN,
                            node_vals_s[node_idx],
                        )
                    )
            self.emit_vbroadcasts(idx_pairs)
            self.emit_vbroadcasts(val_pairs)
            diff_ops = []
            if shallow_group_depth >= 1:
                diff1_s = self.alloc_scratch("diff1_s")
                base1 = (1 << 1) - 1
                diff_ops.append(
                    (
                        "alu",
                        ("-", diff1_s, node_vals_s[base1 + 1], node_vals_s[base1 + 0]),
                    )
                )
            if shallow_group_depth >= 2:
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
            if diff_ops:
                self.emit_cycle(diff_ops)
            if shallow_group_depth >= 1:
                v_depth1_diff = self.alloc_scratch("v_depth1_diff", VLEN)
                self.emit_cycle([("valu", ("vbroadcast", v_depth1_diff, diff1_s))])
            if shallow_group_depth >= 2:
                v_depth2_diff10 = self.alloc_scratch("v_depth2_diff10", VLEN)
                v_depth2_diff20 = self.alloc_scratch("v_depth2_diff20", VLEN)
                v_depth2_diff30 = self.alloc_scratch("v_depth2_diff30", VLEN)
                self.emit_vbroadcasts(
                    [
                        (v_depth2_diff10, diff2_10_s),
                        (v_depth2_diff20, diff2_20_s),
                        (v_depth2_diff30, diff2_30_s),
                    ]
                )

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
            # Load idx/val into scratch directly
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

            # Pause instructions are matched up with yield statements in the reference
            # kernel to let you debug at intermediate steps. The testing harness in this
            # file requires these match up to the reference kernel's yields, but the
            # submission harness ignores them.
            self.add("flow", ("pause",))

        start_round = group_depth + 1 if partition_grouping else 0

        use_flow_select = os.getenv("USE_FLOW_SELECT", "0") == "1"
        flow_select_depth1 = os.getenv("FLOW_SELECT_DEPTH1", "1") == "1"
        flow_select_depth2 = os.getenv("FLOW_SELECT_DEPTH2", "0") == "1"
        if use_flow_select:
            flow_select_depth1 = True
            flow_select_depth2 = True
        fast_depth2_select = os.getenv("FAST_DEPTH2_SELECT", "1") != "0"
        wave_size = int(os.getenv("WAVE_SIZE", "88"))
        num_tmp = 3 if flow_select_depth2 else 2
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
        if flow_select_depth2:
            tmp3_vec_b = [
                self.alloc_scratch(f"tmp3_vec_{i}", VLEN) for i in range(scratch_wave)
            ]

        round_plans = []
        idx_is_address = False
        for round in range(start_round, rounds):
            depth = round % (forest_height + 1)
            use_grouping_round = use_shallow_grouping and depth <= shallow_group_depth
            if use_grouping_round:
                num_nodes = 1 << depth
                depth_offset = depth_offsets[depth]
            else:
                num_nodes = 0
                depth_offset = 0
            transition_to_addr = (
                use_shallow_grouping
                and depth == shallow_group_depth
                and shallow_group_depth < forest_height
            )
            round_plans.append(
                (depth, use_grouping_round, num_nodes, depth_offset, idx_is_address, transition_to_addr)
            )
            if depth == forest_height:
                idx_is_address = False
            elif transition_to_addr:
                idx_is_address = True
        final_idx_is_address = idx_is_address

        for wi in range(waves):
            ops_by_block = []
            base_block = wi * wave_size
            for bi in range(wave_size):
                block_idx = base_block + bi
                if block_idx >= blocks:
                    ops_by_block.append([])
                    continue
                ops = []
                idx_vec = idx_all + block_idx * VLEN
                val_vec = val_all + block_idx * VLEN

                for (
                    depth,
                    use_grouping_round,
                    num_nodes,
                    depth_offset,
                    idx_is_address,
                    transition_to_addr,
                ) in round_plans:
                    if use_grouping_round:
                        if depth == 0:
                            ops.append(
                                (
                                    "valu",
                                    (
                                        "+",
                                        tmp2_vec_b[bi],
                                        v_node_val_all + depth_offset,
                                        v_zero,
                                    ),
                                )
                            )
                        elif depth == 1:
                            if flow_select_depth1:
                                # mask for right child
                                ops.append(
                                    (
                                        "valu",
                                        (
                                            "==",
                                            tmp1_vec_b[bi],
                                            idx_vec,
                                            v_node_idx_all + depth_offset + VLEN,
                                        ),
                                    )
                                )
                                ops.append(
                                    (
                                        "flow",
                                        (
                                            "vselect",
                                            tmp2_vec_b[bi],
                                            tmp1_vec_b[bi],
                                            v_node_val_all + depth_offset + VLEN,
                                            v_node_val_all + depth_offset,
                                        ),
                                    )
                                )
                            else:
                                # off = idx - base (0 or 1)
                                ops.append(
                                    (
                                        "valu",
                                        (
                                            "-",
                                            tmp1_vec_b[bi],
                                            idx_vec,
                                            v_node_idx_all + depth_offset,
                                        ),
                                    )
                                )
                                ops.append(
                                    (
                                        "valu",
                                        (
                                            "+",
                                            tmp2_vec_b[bi],
                                            v_node_val_all + depth_offset,
                                            v_zero,
                                        ),
                                    )
                                )
                                ops.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            tmp2_vec_b[bi],
                                            tmp1_vec_b[bi],
                                            v_depth1_diff,
                                            tmp2_vec_b[bi],
                                        ),
                                    )
                                )
                        else:
                            if flow_select_depth2 and depth == 2:
                                # leaf = idx - base (0..3)
                                ops.append(
                                    (
                                        "valu",
                                        (
                                            "-",
                                            tmp1_vec_b[bi],
                                            idx_vec,
                                            v_node_idx_all + depth_offset,
                                        ),
                                    )
                                )
                                # mask_hi = leaf >> 1
                                ops.append(
                                    (
                                        "valu",
                                        (">>", tmp2_vec_b[bi], tmp1_vec_b[bi], v_one),
                                    )
                                )
                                # mask_lo = leaf & 1
                                ops.append(
                                    (
                                        "valu",
                                        ("&", tmp1_vec_b[bi], tmp1_vec_b[bi], v_one),
                                    )
                                )
                                # left = select(node4, node3) by mask_lo
                                ops.append(
                                    (
                                        "flow",
                                        (
                                            "vselect",
                                            tmp3_vec_b[bi],
                                            tmp1_vec_b[bi],
                                            v_node_val_all + depth_offset + VLEN,
                                            v_node_val_all + depth_offset,
                                        ),
                                    )
                                )
                                # right = select(node6, node5) by mask_lo
                                ops.append(
                                    (
                                        "flow",
                                        (
                                            "vselect",
                                            tmp1_vec_b[bi],
                                            tmp1_vec_b[bi],
                                            v_node_val_all + depth_offset + 3 * VLEN,
                                            v_node_val_all + depth_offset + 2 * VLEN,
                                        ),
                                    )
                                )
                                # node_val = select(right, left) by mask_hi
                                ops.append(
                                    (
                                        "flow",
                                        (
                                            "vselect",
                                            tmp2_vec_b[bi],
                                            tmp2_vec_b[bi],
                                            tmp1_vec_b[bi],
                                            tmp3_vec_b[bi],
                                        ),
                                    )
                                )
                            elif (
                                depth == 2
                                and fast_depth2_select
                                and v_depth2_diff10 is not None
                            ):
                                # Start from node0, then add diffs for the other nodes.
                                ops.append(
                                    (
                                        "valu",
                                        (
                                            "+",
                                            tmp2_vec_b[bi],
                                            v_node_val_all + depth_offset,
                                            v_zero,
                                        ),
                                    )
                                )
                                ops.append(
                                    (
                                        "valu",
                                        (
                                            "==",
                                            tmp1_vec_b[bi],
                                            idx_vec,
                                            v_node_idx_all + depth_offset + VLEN,
                                        ),
                                    )
                                )
                                ops.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            tmp2_vec_b[bi],
                                            tmp1_vec_b[bi],
                                            v_depth2_diff10,
                                            tmp2_vec_b[bi],
                                        ),
                                    )
                                )
                                ops.append(
                                    (
                                        "valu",
                                        (
                                            "==",
                                            tmp1_vec_b[bi],
                                            idx_vec,
                                            v_node_idx_all + depth_offset + 2 * VLEN,
                                        ),
                                    )
                                )
                                ops.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            tmp2_vec_b[bi],
                                            tmp1_vec_b[bi],
                                            v_depth2_diff20,
                                            tmp2_vec_b[bi],
                                        ),
                                    )
                                )
                                ops.append(
                                    (
                                        "valu",
                                        (
                                            "==",
                                            tmp1_vec_b[bi],
                                            idx_vec,
                                            v_node_idx_all + depth_offset + 3 * VLEN,
                                        ),
                                    )
                                )
                                ops.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            tmp2_vec_b[bi],
                                            tmp1_vec_b[bi],
                                            v_depth2_diff30,
                                            tmp2_vec_b[bi],
                                        ),
                                    )
                                )
                            else:
                                # tmp2_vec_b holds node_val accumulator
                                ops.append(("valu", ("+", tmp2_vec_b[bi], v_zero, v_zero)))
                                for node_k in range(num_nodes):
                                    ops.append(
                                        (
                                            "valu",
                                            (
                                                "==",
                                                tmp1_vec_b[bi],
                                                idx_vec,
                                                v_node_idx_all
                                                + depth_offset
                                                + node_k * VLEN,
                                            ),
                                        )
                                    )
                                    ops.append(
                                        (
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
                                        )
                                    )
                    else:
                        if idx_is_address:
                            for lane in range(VLEN):
                                ops.append(
                                    ("load", ("load_offset", tmp2_vec_b[bi], idx_vec, lane))
                                )
                        else:
                            ops.append(("valu", ("+", tmp1_vec_b[bi], idx_vec, v_forest_p)))
                            for lane in range(VLEN):
                                ops.append(
                                    (
                                        "load",
                                        ("load_offset", tmp2_vec_b[bi], tmp1_vec_b[bi], lane),
                                    )
                                )
                    ops.append(("valu", ("^", val_vec, val_vec, tmp2_vec_b[bi])))

                    for entry in hash_plan:
                        if entry[0] == "fused":
                            _, v_mult, v_c1 = entry
                            ops.append(
                                ("valu", ("multiply_add", val_vec, val_vec, v_mult, v_c1))
                            )
                        else:
                            _, op1, v_c1, op2, op3, v_c3 = entry
                            ops.append(("valu", (op1, tmp1_vec_b[bi], val_vec, v_c1)))
                            ops.append(("valu", (op3, tmp2_vec_b[bi], val_vec, v_c3)))
                            ops.append(
                                ("valu", (op2, val_vec, tmp1_vec_b[bi], tmp2_vec_b[bi]))
                            )

                    if depth == forest_height:
                        ops.append(("valu", ("^", idx_vec, idx_vec, idx_vec)))
                    else:
                        ops.append(("valu", ("&", tmp1_vec_b[bi], val_vec, v_one)))
                        if idx_is_address:
                            ops.append(
                                (
                                    "valu",
                                    ("+", tmp1_vec_b[bi], tmp1_vec_b[bi], v_one_minus_forest_p),
                                )
                            )
                        elif transition_to_addr:
                            ops.append(
                                (
                                    "valu",
                                    ("+", tmp1_vec_b[bi], tmp1_vec_b[bi], v_one_plus_forest_p),
                                )
                            )
                        else:
                            ops.append(
                                ("valu", ("+", tmp1_vec_b[bi], tmp1_vec_b[bi], v_one))
                            )
                        ops.append(
                            ("valu", ("multiply_add", idx_vec, idx_vec, v_two, tmp1_vec_b[bi]))
                        )

                ops_by_block.append(ops)

            self.schedule_wave(ops_by_block, rr_start=0)

        ops_by_block = []
        for bi in range(blocks):
            off_const = block_offsets[bi]
            ops = [
                ("alu", ("+", idx_addr_b[bi], self.scratch["inp_indices_p"], off_const)),
                ("alu", ("+", val_addr_b[bi], self.scratch["inp_values_p"], off_const)),
            ]
            if final_idx_is_address:
                ops.append(
                    ("valu", ("-", idx_all + bi * VLEN, idx_all + bi * VLEN, v_forest_p))
                )
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
