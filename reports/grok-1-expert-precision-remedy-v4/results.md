# Expert stacked INT4 + LS channel-alpha multi-block fidelity (#85)

**Agent:** Codex (OpenAI)

**Issue:** GH #85 / Linear RM-608 / beads `goz-3h3`

**Attribution:** Grok supplied only the historical issue-planning lock; Codex (OpenAI) implemented, reviewed, executed, validated, and published this evidence.

**Implementation commit:** `a159d89df1dbe2374b56a1c43ec2b4ac80dbe82e` (`implementation.dirty: false`)

**Runtime:** Python 3.14.6; NumPy 2.5.1

**Embedding SHA-256:** `55ec19a8fdd45960579514bf471e8f5cba24436cdc3f6e9e6bcd3a004ed863f6`

**Run status:** complete; the supervisor validated all three mandatory arms and the canonical decision artifacts emitted after the required final P0 arm

**Protocol:** blocks 0-3, tokens 8192, seed 20260806, `top_k=2`, FP16 control enabled, no 2048 fallback

## Decision

**Option 2 - the selected stacked remedy improves on the re-measured same-budget INT4 baseline, but it still compounds and misses the approximately 0.95 top-1 viability band.**

The canonical winner is **P1 `expert_int4_channel_alpha_123`**, not the plain-INT4 comparator. P1 uses INT4 codes with least-squares per-output-channel alpha on block 0 and FP16 experts on blocks 1-3. Its block-3 top-1 agreement is **0.887329**, block-output cosine is **0.990004**, and chain-exit residual drift is **0.141994**.

The all-block P0 candidate, `expert_int4_channel_alpha`, is lower complexity but ranks second on the locked viability/top-1/cosine/exit-drift tuple. The same-budget `expert_int4` arm is present only as the baseline comparator and is excluded from the candidate set.

### Locked ranking contract

- Required evidence: baseline `expert_int4`, P1 `expert_int4_channel_alpha_123`, and P0 `expert_int4_channel_alpha`; all three are present, complete, and valid.
- Canonical candidates: exactly P1 `expert_int4_channel_alpha_123` and P0 `expert_int4_channel_alpha`.
- Comparator only: `expert_int4`.
- Ordered candidates: P1, then P0.
- Winner: P1 `expert_int4_channel_alpha_123`.
- Tie-break: not needed. P1 has the better locked metric rank; an exact tie would have selected lower-complexity P0.
- Missing arms: none. Invalid arms: none. Global validation errors: none.

### Why Option 2

- P1 improves over re-measured INT4 by **+0.063599** block-3 top-1, **+0.009438** block-3 cosine, and **0.055728** lower chain-exit drift.
- P1 is nevertheless not viable under the locked band: block-3 top-1 is **0.887329**, below approximately **0.95**, and its chain is classified `superlinear_or_runaway`.
- P0 does not beat plain INT4: it changes block-3 top-1 by **-0.021973**, block-3 cosine by **-0.004384**, and chain-exit drift by **+0.023387** (worse).
- Option 1 is therefore not selected because neither ranked candidate restores viability.
- Option 3 is not selected because P1 clearly improves on the same-budget INT4 comparator.
- Option 4 is not selected because all three required arms are fresh, complete, comparable, and provenance-valid.

## Same-budget comparison

All values below are measured at 8192 tokens with the same seed, blocks, `top_k`, packs, NPY inputs, clean implementation commit, Python/NumPy runtime, and pinned embedding content.

| Signal | INT4 baseline (comparator) | P1 alpha+HP123 | P0 alpha all blocks |
|---|---:|---:|---:|
| block-3 top-1 | 0.823730 | **0.887329** | 0.801758 |
| block-3 top-2 | 0.721069 | **0.826050** | 0.686401 |
| block-3 block-output cosine | 0.980566 | **0.990004** | 0.976182 |
| chain-exit residual drift | 0.197722 | **0.141994** | 0.221109 |
| viable | false | false | false |
| compounding | superlinear/runaway | superlinear/runaway | superlinear/runaway |
| ranking role | comparator only | **candidate, rank 1** | candidate, rank 2 |

Historical issue #80 P0 all-INT4 reported block-3 top-1 **0.850586 at 2048 tokens**. It is cited for context only: it is not same-budget evidence, not part of this ranking, and was not used as a fallback.

## Method

- A standard-library-only out-of-process supervisor launched exactly one arm at a time in the locked order: baseline -> P1 -> P0.
- Before touching an output tree, the supervisor acquired a blocking, persistent sibling POSIX `flock` and held it across the full three-arm transaction, validation, and final publication. Concurrent same-output supervisors therefore serialize; an interrupted waiter exits with status 6 without modifying the active output.
- The supervisor hard-locked tokens 8192, seed 20260806, blocks `0,1,2,3`, `top_k=2`, and FP16 controls. `fallback_tokens` is `null`; no 2048-token command was launched.
- The supervisor hashed the entire 3,221,225,600-byte embedding shard once in approximately 64 MiB chunks, pinned the digest into every child artifact, and verified both the path and opened target identity before and after the run.
- Each child fingerprinted every NPY source directory before source loading and again after all forwards. The post-forward fingerprint had to equal the pre-forward fingerprint before provenance or canonical evidence could be accepted.
- Baseline and P1 wrote evidence-only `metrics.json` files. P0 consumed both as `--comparison-metrics`, assembled the decision, and owned the canonical `metrics.json` and generated `results.md` skeleton.
- The supervisor independently recomputed complete summaries, candidate ranking, baseline deltas, tie behavior, and the decision from the three raw chains before accepting the P0 artifacts.
- The supervisor required the same implementation SHA, clean-tree state, embedding digest, architecture source, Python version, and NumPy version across all arms.
- Each arm used a sequential, paired residual trajectory. No Gaussian proxy was used, and embedding input was not substituted for blocks greater than 0.
- Only expert payload precision changed. Attention, routers, norms, and the FP reference remained high precision.
- Plain INT4 uses persisted per-output-channel absmax codes/scales. The alpha arms reuse the shared INT4 codes with float64 least-squares per-output-channel alpha scales.
- Expert tensors were traversed deterministically in approximately 64 MiB whole-row intra-expert chunks. Reference fingerprints streamed in C order; quantization used a two-pass absmax build and read-only memory-mapped code reuse.
- Cache publication uses a persistent per-block POSIX advisory lock shared by absmax and LS-alpha modes, atomic replacement, file `fsync`, and strict parent-directory `fsync`; active-sidecar removal is also directory-synced. The cache therefore requires a local POSIX filesystem with working `flock` and directory `fsync` semantics.
- The external side-table cache was implementation-scoped and fingerprint-validated. This exact-head run reused entries produced during an interrupted earlier attempt; directory `gh85-v4-8192-int4-side-ce90d12` finished at 19,345,718,032 bytes and is intentionally outside the tracked report tree. No earlier scientific output was promoted: Codex (OpenAI) regenerated and validated all canonical tracked artifacts from clean `a159d89`.

## Per-block evidence

### Baseline comparator - `expert_int4`

INT4 absmax experts are active on every measured block.

| block | block-output cos | residual-in drift | top-1 | top-2 | JS bits | MoE-output cos |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.998637 | 0.000000 | 1.000000 | 1.000000 | 0 | 0.994210 |
| 1 | 0.997567 | 0.052476 | 0.982300 | 0.938599 | 2.03307e-05 | 0.988009 |
| 2 | 0.988657 | 0.069880 | 0.926270 | 0.886108 | 0.000194974 | 0.961282 |
| 3 | 0.980566 | 0.150499 | 0.823730 | 0.721069 | 0.000317050 | 0.939914 |

Chain-exit residual cosine: **0.980566**. Chain-exit residual drift: **0.197722**.

### P1 - `expert_int4_channel_alpha_123`

Block 0 uses shared INT4 codes with LS channel-alpha; blocks 1-3 use FP16 experts.

| block | block-output cos | residual-in drift | top-1 | top-2 | JS bits | MoE-output cos |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.998136 | 0.000000 | 1.000000 | 1.000000 | 0 | 0.993498 |
| 1 | 0.998285 | 0.061249 | 0.982056 | 0.932861 | 3.42763e-05 | 0.996178 |
| 2 | 0.993960 | 0.058675 | 0.935181 | 0.906616 | 0.000200284 | 0.981521 |
| 3 | 0.990004 | 0.109763 | 0.887329 | 0.826050 | 7.14824e-05 | 0.970410 |

Chain-exit residual cosine: **0.990004**. Chain-exit residual drift: **0.141994**.

### P0 - `expert_int4_channel_alpha`

Shared INT4 codes with LS channel-alpha scales are active on all four measured blocks.

| block | block-output cos | residual-in drift | top-1 | top-2 | JS bits | MoE-output cos |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.998136 | 0.000000 | 1.000000 | 1.000000 | 0 | 0.993498 |
| 1 | 0.997071 | 0.061249 | 0.982056 | 0.932861 | 3.42763e-05 | 0.986153 |
| 2 | 0.985698 | 0.076533 | 0.911011 | 0.868164 | 0.000341646 | 0.950916 |
| 3 | 0.976182 | 0.168773 | 0.801758 | 0.686401 | 0.000723405 | 0.925091 |

Chain-exit residual cosine: **0.976182**. Chain-exit residual drift: **0.221109**.

### FP16 control

The deterministic FP16 control trajectory is identical across the three arms and passed supervisor validation.

| block | block-output cos | residual-in drift | top-1 | top-2 | MoE-output cos |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.999966 | 0.000000 | 0.999146 | 0.996582 | 0.999896 |
| 1 | 0.999966 | 0.008218 | 0.999634 | 0.998169 | 0.999901 |
| 2 | 0.999765 | 0.008255 | 0.996826 | 0.995361 | 0.999119 |
| 3 | 0.999437 | 0.021683 | 0.991211 | 0.987549 | 0.998092 |

## Provenance

The supervisor ran from 2026-08-24 14:05:32.377964Z through 17:13:18.488727Z (3:07:46.110763). It recorded the clean implementation identity before launch, hashed and pinned the entire embedding shard, revalidated protocol and provenance after every child, and independently validated the P0/P1 ranking after the final arm. The exact-head rerun was hosted by the user service manager so a chat or terminal reset could not terminate the scientific process tree.

| block | GOZ1 pack SHA-256 | NPY directory SHA-256 |
|---:|---|---|
| 0 | `eb28728b6b66454753bc67072b105e701d99f589ca20f4edf174fecd06e0e1c4` | `42ccd9c9a98a6a222d95a157f72e49c0364624f89074c7a7a381cf3cdcf6aa9b` |
| 1 | `dd66181d6ce42ed47a6a257feaf606d20c4b0b23f471004ae373b21c1398c17f` | `fada26dbc8f2c330d01f9e6b18c4384d35850e55c52ae1b5162885a9d7cdd181` |
| 2 | `e61641e19735293e6802c33d69dda6f83507480fc955141f358eb0df31da8560` | `5e9a15c0de698645f82491a1b3e5118f1230f39751fc513303dffb0067f55c7f` |
| 3 | `9db504ff9ee08a2523f74e3f842228296458fc524a1ccf406de46c53d9e18302` | `39322c3f8ae0f13313439faa8dd5a54edc102310cb0f8c03cf131b87bf5909e7` |

The launch snapshot recorded 64,906,240,000 bytes total RAM, 27,845,763,072 bytes available RAM, and 106,196,992 bytes free swap. No launch gate was applied. The end snapshot recorded 23,424,364,544 bytes available RAM and a maximum child RSS of 15,706,260 KiB. The user service manager observed a 23.2 GiB process-tree memory peak and 10.1 MiB swap peak. These are observational host snapshots, not proof of behavior on another host.

## Reproduction

Implementation must be committed and clean before launch. Configure local input paths, then run only the supervisor:

```bash
GH85_NPY_ROOT=/path/to/export-npy
GH85_PACK_ROOT=/path/to/multiblock-68
GH85_EMBEDDING=/path/to/embedding__slot_00__token_embedding.npy
GH85_SIDE_ROOT=/path/to/cache/gh85-v4-8192-int4-side-ce90d12

python3 scripts/grok1_multiblock_v4_supervisor.py \
  --npy-root "$GH85_NPY_ROOT" \
  --npy-pattern 'goz68-block_{block:03d}-attn' \
  --pack-root "$GH85_PACK_ROOT" \
  --pack-pattern 'block_{block:03d}-attention_plus_expert.goz1' \
  --embedding-shard "$GH85_EMBEDDING" \
  --int4-side-root "$GH85_SIDE_ROOT" \
  --out reports/grok-1-expert-precision-remedy-v4
```

The supervisor owns the baseline -> P1 -> P0 launch order and fails closed to Option 4 on a child failure, timeout, invalid/stale evidence, protocol mismatch, or provenance mismatch. Its persistent sibling output lock serializes concurrent invocations that target the same resolved output directory. Do not replace it with three direct experiment commands. Persistent child logs redact the interpreter and absolute input paths while preserving the actual executed argument order.

## Artifacts

`metrics.json` is the machine source of truth. The two nested `metrics.json` files are evidence-only inputs and intentionally contain no independent decision. Progress files are atomic block snapshots; `supervisor-progress.json` is the terminal arm/protocol record. Logs are bounded captured child output. The large reusable side tables remain external.

```text
reports/grok-1-expert-precision-remedy-v4/
|-- host-at-launch.json
|-- host-at-end.json
|-- supervisor-progress.json
|-- progress-int4-baseline.json
|-- progress-int4-channel-alpha-123.json
|-- progress-int4-channel-alpha.json
|-- run-01-baseline.log
|-- run-02-p1.log
|-- run-03-p0.log
|-- int4-baseline/metrics.json
|-- int4-channel-alpha-123/metrics.json
|-- metrics.json
`-- results.md
```
