# Grok-1 ordered text inputs v1

This interface is the implementation-only portion of GH #127 / RM-1026. It
freezes and validates inputs; it does **not** provide real Grok-1 tokenizer
artifacts, select experimental arms, run model weights, or publish the two
required 8192-token comparisons.

## Production contract

Create a JSON manifest with `schema_version: "grok1-ordered-inputs-v1"`,
`input_mode: "ordered_text"`, `fixture: false`, `selection_seed: 20260806`, and
`window_tokens: 8192`. The `tokenizer` object must pin a non-empty identity,
revision, vocabulary size, special-token/BOS/EOS policy, and a portable relative
model-file path plus its SHA-256. The `source` object similarly pins identity,
revision, license, extraction/normalization version, and the source artifact's
relative path and SHA-256.

Exactly one `calibration` and two `held_out` windows are accepted. Each window
pins a unique ID, document ID, source range (`start`, exclusive `end`),
`sequence_boundary: "independent"`, a relative `.npy` token path, token count,
whole-file SHA-256, order-sensitive token SHA-256, and order-independent
multiset/content SHA-256. Arrays must be one-dimensional integer arrays with
every ID in the declared vocabulary. The loader returns owned, read-only int64
arrays without sorting or deduplication, so causal order and repeats survive.

Validation fails closed on digest changes, non-integer/out-of-range IDs,
overlapping ranges in a document, cross-partition range leakage, identical
token multisets, short/missing partitions, non-independent boundaries, absolute
or parent-traversing paths, and sampled/mixed input modes. The content check is
exact-token-multiset equality only; it does not detect semantic, substring, or
general near duplicates. Window producers must perform and document stronger
near-duplicate review before freezing a production manifest.

Validate a frozen manifest before passing its returned arrays to an experiment:

```bash
python3 scripts/grok1_ordered_inputs.py path/to/manifest.json
```

Do not use `--allow-fixture` for evidence. That switch exists solely for the
visibly labelled synthetic unit fixtures. A production manifest must reference
the pinned real tokenizer revision/model digest and licensed source artifact;
the repository's fixtures make no claim of Grok-1 tokenizer compatibility.

The module does not alter `grok1_block0_experiment.token_ids` or the multi-block
call to it. Historical sampled-ID evidence therefore remains sorted,
without-replacement, and separately labelled; it must not be mixed into a text
window result table.

## Reference-forward evidence boundary

`scripts/test_grok1_block_forward.py` remains the reused small reference surface.
It checks role/shape mapping, embedding scaling, RMSNorm, RoPE and causal masking,
attention tensor orientation against independently written naive calculations,
GeGLU branch placement, top-k routing, and expert combination. These are tiny
kernel/invariant tests. They do not compare this Python implementation against a
pinned upstream Grok-1 runtime, and neither they nor FP16-versus-FP32 agreement
establish upstream parity. No new parity claim is made by this input loader.

## Still outstanding for the experiment

After RM-1024 selects its matched arm, freeze the exact licensed production
source, real tokenizer, three windows, arm list, metrics, and thresholds before
examining held-out results. Then run the high-precision reference, FP16 control,
plain all-block absmax INT4, and selected comparison on blocks 0–3, reporting
both held-out windows separately and aggregate variability. Missing weights or
resources defer that measurement; short fixtures are never a scientific
fallback.
