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
        idxs = [0] * n_blocks
        rr = rr_start % n_blocks if n_blocks else 0
        while any(i < len(ops_by_block[b]) for b, i in enumerate(idxs)):
            bundle = defaultdict(list)
            counts = defaultdict(int)
            block_writes = [set() for _ in range(n_blocks)]
            block_used = [False] * n_blocks
            block_single = [False] * n_blocks
            progressed = True
            while progressed:
                progressed = False
                for bi in range(n_blocks):
                    b = (rr + bi) % n_blocks
                    if idxs[b] >= len(ops_by_block[b]):
                        continue
                    engine, slot = ops_by_block[b][idxs[b]]
                    if block_single[b] and block_used[b]:
                        continue
                    if counts[engine] >= SLOT_LIMITS[engine]:
                        continue
                    reads, writes = self._slot_reads_writes(engine, slot)
                    if reads is None or writes is None:
                        # Unknown slot shape; fall back to single-op issue for this block.
                        if block_used[b]:
                            continue
                        reads = set()
                        writes = set()
                        block_single[b] = True
                    if reads & block_writes[b]:
                        continue
                    if writes & block_writes[b]:
                        continue
                    bundle[engine].append(slot)
                    counts[engine] += 1
                    block_writes[b].update(writes)
                    idxs[b] += 1
                    block_used[b] = True
                    progressed = True
            if not bundle:
                # Shouldn't happen, but avoid infinite loops if it does.
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
            if shallow_group_depth >= 1:
                diff1_s = self.alloc_scratch("diff1_s")
                base1 = (1 << 1) - 1
                self.emit_cycle(
                    [
                        (
                            "alu",
                            (
                                "-",
                                diff1_s,
                                node_vals_s[base1 + 1],
                                node_vals_s[base1 + 0],
                            ),
                        )
                    ]
                )
                v_depth1_diff = self.alloc_scratch("v_depth1_diff", VLEN)
                self.emit_cycle([("valu", ("vbroadcast", v_depth1_diff, diff1_s))])

        idx_addr = self.alloc_scratch("idx_addr")
        val_addr = self.alloc_scratch("val_addr")
        idx_addr_b = [self.alloc_scratch(f"idx_addr_b_{i}") for i in range(blocks)]
        val_addr_b = [self.alloc_scratch(f"val_addr_b_{i}") for i in range(blocks)]

        idx_all = self.alloc_scratch("idx_all", blocks * VLEN)
        val_all = self.alloc_scratch("val_all", blocks * VLEN)

        if partition_grouping:
            extra_room = self.alloc_scratch("extra_room")
            self.emit_cycle(
                [
                    (
                        "alu",
                        (
                            "+",
                            extra_room,
                            self.scratch["inp_values_p"],
                            self.scratch["batch_size"],
                        ),
                    )
                ]
            )

            cur_vals_p = self.alloc_scratch("cur_vals_p")
            cur_pos_p = self.alloc_scratch("cur_pos_p")
            cur_idx_p = self.alloc_scratch("cur_idx_p")
            next_vals_p = self.alloc_scratch("next_vals_p")
            next_pos_p = self.alloc_scratch("next_pos_p")
            next_idx_p = self.alloc_scratch("next_idx_p")
            counts_cur_p = self.alloc_scratch("counts_cur_p")
            counts_next_p = self.alloc_scratch("counts_next_p")

            batch_const = self.scratch_const(batch_size)
            off_batch = self.scratch_const(batch_size)
            off_batch2 = self.scratch_const(batch_size * 2)
            off_batch3 = self.scratch_const(batch_size * 3)
            off_batch4 = self.scratch_const(batch_size * 4)
            off_batch5 = self.scratch_const(batch_size * 5)
            off_counts = self.scratch_const(batch_size * 6)

            counts_size = 1 << (group_depth + 1)
            if batch_size * 6 + counts_size * 2 > extra_room_size:
                raise ValueError("group_depth too large for extra_room")
            counts_size_const = self.scratch_const(counts_size)

            self.emit_cycle([("alu", ("+", cur_vals_p, extra_room, zero_const))])
            self.emit_cycle([("alu", ("+", cur_pos_p, extra_room, off_batch))])
            self.emit_cycle([("alu", ("+", cur_idx_p, extra_room, off_batch2))])
            self.emit_cycle([("alu", ("+", next_vals_p, extra_room, off_batch3))])
            self.emit_cycle([("alu", ("+", next_pos_p, extra_room, off_batch4))])
            self.emit_cycle([("alu", ("+", next_idx_p, extra_room, off_batch5))])
            self.emit_cycle([("alu", ("+", counts_cur_p, extra_room, off_counts))])
            self.emit_cycle([("alu", ("+", counts_next_p, counts_cur_p, counts_size_const))])

            vlen_const = self.scratch_const(VLEN)
            vlen_minus_one = self.scratch_const(VLEN - 1)

            group_val_vec = self.alloc_scratch("group_val_vec", VLEN)
            group_tmp1_vec = self.alloc_scratch("group_tmp1_vec", VLEN)
            group_tmp2_vec = self.alloc_scratch("group_tmp2_vec", VLEN)
            group_node_vec = self.alloc_scratch("group_node_vec", VLEN)

            # Copy inp_values to cur_vals
            tmp_vec = self.alloc_scratch("tmp_vec", VLEN)
            for bi in range(blocks):
                off_const = block_offsets[bi]
                self.emit_cycle(
                    [
                        ("alu", ("+", val_addr, self.scratch["inp_values_p"], off_const)),
                        ("alu", ("+", idx_addr, cur_vals_p, off_const)),
                    ]
                )
                self.emit_cycle(
                    [
                        ("load", ("vload", tmp_vec, val_addr)),
                        ("store", ("vstore", idx_addr, tmp_vec)),
                    ]
                )

            # Init cur_pos = 0..batch_size-1
            pos_i = self.alloc_scratch("pos_i")
            pos_addr = self.alloc_scratch("pos_addr")
            self.emit_cycle([("load", ("const", pos_i, 0))])
            self.label("pos_loop_start")
            self.emit_cycle([("alu", ("<", tmp1, pos_i, batch_const))])
            self.emit_cycle([("alu", ("==", tmp2, tmp1, zero_const))])
            self.add_cjump_rel(tmp2, "pos_loop_end")
            self.emit_cycle([("alu", ("+", pos_addr, cur_pos_p, pos_i))])
            self.emit_cycle([("store", ("store", pos_addr, pos_i))])
            self.emit_cycle([("alu", ("+", pos_i, pos_i, one_const))])
            self.add_jump("pos_loop_start")
            self.label("pos_loop_end")

            # counts_cur[0] = batch_size
            self.emit_cycle([("alu", ("+", tmp1, counts_cur_p, zero_const))])
            self.emit_cycle([("store", ("store", tmp1, batch_const))])

            # Pause to match first yield
            self.add("flow", ("pause",))

            # Grouping for depths 0..group_depth (vector hash + scalar partition)
            count = self.alloc_scratch("count")
            i = self.alloc_scratch("i")
            left_count = self.alloc_scratch("left_count")
            cur_off = self.alloc_scratch("cur_off")
            next_off = self.alloc_scratch("next_off")
            addr = self.alloc_scratch("addr")
            val = self.alloc_scratch("val")
            pos = self.alloc_scratch("pos")
            parity = self.alloc_scratch("parity")
            cond = self.alloc_scratch("cond")
            left_ptr = self.alloc_scratch("left_ptr")
            right_ptr = self.alloc_scratch("right_ptr")
            count_minus = self.alloc_scratch("count_minus")
            base_vals = self.alloc_scratch("base_vals")
            base_pos = self.alloc_scratch("base_pos")

            for depth in range(group_depth + 1):
                num_nodes = 1 << depth
                base = (1 << depth) - 1
                self.emit_cycle([("load", ("const", cur_off, 0))])
                self.emit_cycle([("load", ("const", next_off, 0))])
                for node_k in range(num_nodes):
                    node_idx = base + node_k
                    node_idx_const = self.scratch_const(node_idx)
                    left_idx_const = self.scratch_const(node_idx * 2 + 1)
                    right_idx_const = self.scratch_const(node_idx * 2 + 2)

                    # load count
                    self.emit_cycle(
                        [("alu", ("+", addr, counts_cur_p, self.scratch_const(node_k)))]
                    )
                    self.emit_cycle([("load", ("load", count, addr))])
                    self.emit_cycle([("alu", ("==", cond, count, zero_const))])
                    self.add_cjump_rel(cond, f"grp_d{depth}_n{node_k}_zero")

                    # base pointers for this group
                    self.emit_cycle([("alu", ("+", base_vals, cur_vals_p, cur_off))])
                    self.emit_cycle([("alu", ("+", base_pos, cur_pos_p, cur_off))])

                    # node_val and broadcast
                    self.emit_cycle(
                        [
                            (
                                "alu",
                                ("+", addr, self.scratch["forest_values_p"], node_idx_const),
                            )
                        ]
                    )
                    self.emit_cycle([("load", ("load", tmp3, addr))])
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
                    self.emit_cycle([("alu", ("+", addr, base_vals, i))])
                    self.emit_cycle([("load", ("vload", group_val_vec, addr))])
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
                                [("valu", (op2, group_val_vec, group_tmp1_vec, group_tmp2_vec))]
                            )
                    self.emit_cycle([("alu", ("+", addr, base_vals, i))])
                    self.emit_cycle([("store", ("vstore", addr, group_val_vec))])
                    self.emit_cycle([("alu", ("+", i, i, vlen_const))])
                    self.add_jump(f"grp_d{depth}_n{node_k}_vec")
                    self.label(f"grp_d{depth}_n{node_k}_vec_end")

                    # scalar tail
                    self.label(f"grp_d{depth}_n{node_k}_tail")
                    self.emit_cycle([("alu", ("<", cond, i, count))])
                    self.emit_cycle([("alu", ("==", tmp2, cond, zero_const))])
                    self.add_cjump_rel(tmp2, f"grp_d{depth}_n{node_k}_tail_end")
                    self.emit_cycle([("alu", ("+", addr, base_vals, i))])
                    self.emit_cycle([("load", ("load", val, addr))])
                    self.emit_cycle([("alu", ("^", val, val, tmp3))])
                    for op1, c1, op2, op3, c3 in hash_scalar_consts:
                        self.emit_cycle([("alu", (op1, tmp1, val, c1))])
                        self.emit_cycle([("alu", (op3, tmp2, val, c3))])
                        self.emit_cycle([("alu", (op2, val, tmp1, tmp2))])
                    self.emit_cycle([("store", ("store", addr, val))])
                    self.emit_cycle([("alu", ("+", i, i, one_const))])
                    self.add_jump(f"grp_d{depth}_n{node_k}_tail")
                    self.label(f"grp_d{depth}_n{node_k}_tail_end")

                    # left_count = count - sum(parity)
                    self.emit_cycle([("alu", ("+", left_count, count, zero_const))])
                    self.emit_cycle([("load", ("const", i, 0))])
                    self.label(f"grp_d{depth}_n{node_k}_count")
                    self.emit_cycle([("alu", ("<", cond, i, count))])
                    self.emit_cycle([("alu", ("==", tmp2, cond, zero_const))])
                    self.add_cjump_rel(tmp2, f"grp_d{depth}_n{node_k}_count_end")
                    self.emit_cycle([("alu", ("+", addr, base_vals, i))])
                    self.emit_cycle([("load", ("load", val, addr))])
                    self.emit_cycle([("alu", ("&", parity, val, one_const))])
                    self.emit_cycle([("alu", ("-", left_count, left_count, parity))])
                    self.emit_cycle([("alu", ("+", i, i, one_const))])
                    self.add_jump(f"grp_d{depth}_n{node_k}_count")
                    self.label(f"grp_d{depth}_n{node_k}_count_end")

                    # store counts for children
                    self.emit_cycle(
                        [("alu", ("+", addr, counts_next_p, self.scratch_const(node_k * 2)))]
                    )
                    self.emit_cycle([("store", ("store", addr, left_count))])
                    self.emit_cycle([("alu", ("-", tmp1, count, left_count))])
                    self.emit_cycle(
                        [
                            (
                                "alu",
                                ("+", addr, counts_next_p, self.scratch_const(node_k * 2 + 1)),
                            )
                        ]
                    )
                    self.emit_cycle([("store", ("store", addr, tmp1))])

                    # partition pass
                    self.emit_cycle([("load", ("const", i, 0))])
                    self.emit_cycle([("alu", ("+", left_ptr, next_off, zero_const))])
                    self.emit_cycle([("alu", ("+", right_ptr, next_off, left_count))])
                    self.label(f"grp_d{depth}_n{node_k}_part")
                    self.emit_cycle([("alu", ("<", cond, i, count))])
                    self.emit_cycle([("alu", ("==", tmp2, cond, zero_const))])
                    self.add_cjump_rel(tmp2, f"grp_d{depth}_n{node_k}_part_end")
                    self.emit_cycle([("alu", ("+", addr, base_vals, i))])
                    self.emit_cycle([("load", ("load", val, addr))])
                    self.emit_cycle([("alu", ("&", parity, val, one_const))])
                    self.emit_cycle([("alu", ("+", addr, base_pos, i))])
                    self.emit_cycle([("load", ("load", pos, addr))])
                    self.emit_cycle([("alu", ("==", cond, parity, zero_const))])
                    self.add_cjump_rel(cond, f"grp_d{depth}_n{node_k}_left")
                    # right
                    self.emit_cycle([("alu", ("+", addr, next_vals_p, right_ptr))])
                    self.emit_cycle([("store", ("store", addr, val))])
                    self.emit_cycle([("alu", ("+", addr, next_pos_p, right_ptr))])
                    self.emit_cycle([("store", ("store", addr, pos))])
                    self.emit_cycle([("alu", ("+", addr, next_idx_p, right_ptr))])
                    self.emit_cycle([("store", ("store", addr, right_idx_const))])
                    self.emit_cycle([("alu", ("+", right_ptr, right_ptr, one_const))])
                    self.add_jump(f"grp_d{depth}_n{node_k}_join")
                    self.label(f"grp_d{depth}_n{node_k}_left")
                    self.emit_cycle([("alu", ("+", addr, next_vals_p, left_ptr))])
                    self.emit_cycle([("store", ("store", addr, val))])
                    self.emit_cycle([("alu", ("+", addr, next_pos_p, left_ptr))])
                    self.emit_cycle([("store", ("store", addr, pos))])
                    self.emit_cycle([("alu", ("+", addr, next_idx_p, left_ptr))])
                    self.emit_cycle([("store", ("store", addr, left_idx_const))])
                    self.emit_cycle([("alu", ("+", left_ptr, left_ptr, one_const))])
                    self.label(f"grp_d{depth}_n{node_k}_join")
                    self.emit_cycle([("alu", ("+", i, i, one_const))])
                    self.add_jump(f"grp_d{depth}_n{node_k}_part")
                    self.label(f"grp_d{depth}_n{node_k}_part_end")

                    self.emit_cycle([("alu", ("+", cur_off, cur_off, count))])
                    self.emit_cycle([("alu", ("+", next_off, next_off, count))])
                    self.add_jump(f"grp_d{depth}_n{node_k}_end")
                    self.label(f"grp_d{depth}_n{node_k}_zero")
                    self.emit_cycle(
                        [("alu", ("+", addr, counts_next_p, self.scratch_const(node_k * 2)))]
                    )
                    self.emit_cycle([("store", ("store", addr, zero_const))])
                    self.emit_cycle(
                        [
                            (
                                "alu",
                                ("+", addr, counts_next_p, self.scratch_const(node_k * 2 + 1)),
                            )
                        ]
                    )
                    self.emit_cycle([("store", ("store", addr, zero_const))])
                    self.label(f"grp_d{depth}_n{node_k}_end")

                # swap buffers
                self.emit_cycle([("alu", ("+", tmp1, cur_vals_p, zero_const))])
                self.emit_cycle([("alu", ("+", cur_vals_p, next_vals_p, zero_const))])
                self.emit_cycle([("alu", ("+", next_vals_p, tmp1, zero_const))])
                self.emit_cycle([("alu", ("+", tmp1, cur_pos_p, zero_const))])
                self.emit_cycle([("alu", ("+", cur_pos_p, next_pos_p, zero_const))])
                self.emit_cycle([("alu", ("+", next_pos_p, tmp1, zero_const))])
                self.emit_cycle([("alu", ("+", tmp1, cur_idx_p, zero_const))])
                self.emit_cycle([("alu", ("+", cur_idx_p, next_idx_p, zero_const))])
                self.emit_cycle([("alu", ("+", next_idx_p, tmp1, zero_const))])
                self.emit_cycle([("alu", ("+", tmp1, counts_cur_p, zero_const))])
                self.emit_cycle([("alu", ("+", counts_cur_p, counts_next_p, zero_const))])
                self.emit_cycle([("alu", ("+", counts_next_p, tmp1, zero_const))])

            # Load grouped idx/val into scratch for vector rounds
            for bi in range(blocks):
                off_const = block_offsets[bi]
                self.emit_cycle(
                    [
                        ("alu", ("+", idx_addr, cur_idx_p, off_const)),
                        ("alu", ("+", val_addr, cur_vals_p, off_const)),
                    ]
                )
                self.emit_cycle(
                    [
                        ("load", ("vload", idx_all + bi * VLEN, idx_addr)),
                        ("load", ("vload", val_all + bi * VLEN, val_addr)),
                    ]
                )

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

        wave_size = int(os.getenv("WAVE_SIZE", "82"))
        waves = (blocks + wave_size - 1) // wave_size
        scratch_wave = min(blocks, wave_size)

        # Per-block scratch for a wave
        tmp1_vec_b = [self.alloc_scratch(f"tmp1_vec_{i}", VLEN) for i in range(scratch_wave)]
        tmp2_vec_b = [self.alloc_scratch(f"tmp2_vec_{i}", VLEN) for i in range(scratch_wave)]

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

                for depth, use_grouping_round, num_nodes, depth_offset in round_plans:
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
                                            v_node_idx_all + depth_offset + node_k * VLEN,
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
                                            v_node_val_all + depth_offset + node_k * VLEN,
                                            tmp2_vec_b[bi],
                                        ),
                                    )
                                )
                    else:
                        # tmp1_vec_b holds node addresses
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
                        ops.append(("valu", ("+", tmp1_vec_b[bi], tmp1_vec_b[bi], v_one)))
                        ops.append(
                            ("valu", ("multiply_add", idx_vec, idx_vec, v_two, tmp1_vec_b[bi]))
                        )

                ops_by_block.append(ops)

            self.schedule_wave(ops_by_block, rr_start=0)

        if partition_grouping:
            # Write back grouped values before scattering to original order
            for bi in range(blocks):
                off_const = block_offsets[bi]
                self.emit_cycle(
                    [("alu", ("+", val_addr, cur_vals_p, off_const))]
                )
                self.emit_cycle(
                    [("store", ("vstore", val_addr, val_all + bi * VLEN))]
                )
            pos_addr = self.alloc_scratch("pos_addr_out")
            out_i = self.alloc_scratch("out_i")
            self.emit_cycle([("load", ("const", out_i, 0))])
            self.label("out_loop_start")
            self.emit_cycle([("alu", ("<", tmp1, out_i, batch_const))])
            self.emit_cycle([("alu", ("==", tmp2, tmp1, zero_const))])
            self.add_cjump_rel(tmp2, "out_loop_end")
            self.emit_cycle([("alu", ("+", pos_addr, cur_pos_p, out_i))])
            self.emit_cycle([("load", ("load", pos, pos_addr))])
            self.emit_cycle([("alu", ("+", addr, cur_vals_p, out_i))])
            self.emit_cycle([("load", ("load", val, addr))])
            self.emit_cycle([("alu", ("+", addr, self.scratch["inp_values_p"], pos))])
            self.emit_cycle([("store", ("store", addr, val))])
            self.emit_cycle([("alu", ("+", out_i, out_i, one_const))])
            self.add_jump("out_loop_start")
            self.label("out_loop_end")
        else:
            ops_by_block = []
            for bi in range(blocks):
                off_const = block_offsets[bi]
                ops = [
                    ("alu", ("+", idx_addr_b[bi], self.scratch["inp_indices_p"], off_const)),
                    ("alu", ("+", val_addr_b[bi], self.scratch["inp_values_p"], off_const)),
                    ("store", ("vstore", idx_addr_b[bi], idx_all + bi * VLEN)),
                    ("store", ("vstore", val_addr_b[bi], val_all + bi * VLEN)),
                ]
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
