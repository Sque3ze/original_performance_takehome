# Score Tracker

All measurements are from `python tests/submission_tests.py` as required by the readme.

| Date (YYYY-MM-DD) | Change | Cycles | Notes |
| --- | --- | --- | --- |
| 2026-01-22 | Baseline (no code changes) | 147734 | Initial run; expected to fail speed thresholds. |
| 2026-01-22 | Phase 1: removed `flow select` for parity and wrap (ALU-only) | 143638 | Speedup ~1.0285× over baseline. |
| 2026-01-22 | Phase 2+3: SIMD vload/vstore + basic VLIW bundling (no cross-block pipeline) | 13376 | Vectorized blocks of 8; packed multiple slots per cycle. |
| 2026-01-22 | Phase 2+3b: wave scheduler across 8 blocks (VLIW list scheduling) | 4928 | Interleaved ops across blocks to fill slot limits. |
| 2026-01-22 | Phase 4: idx/val resident + hash/idx multiply_add + leaf specialization + shallow grouping (depth<=4) | 3453 | Load pressure reduced; valu became main bottleneck. |
| 2026-01-22 | Disabled partition grouping (extra_room invalid); current vector wave path | 3578 | Tests pass; regression vs shallow grouping due to removed depth<=4 selection path. |
| 2026-01-22 | Shallow grouping via per-round prebroadcast (depth<=4) | 3264 | Reduced load pressure with fewer VALU ops; new best. |
