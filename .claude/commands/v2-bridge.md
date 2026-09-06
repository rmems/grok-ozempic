# /v2-bridge — verify the V2 structural manifest bridge (GH #40 / RM-191, **shipped**)

**This work is done.** #40 / RM-191 landed in `67034e8` ("feat: V2 structural name
bridge for resolve_manifest (#40 / RM-191) (#55)"). This command is now a
*verification* recipe, not a work order.

Until GH #96 this file still described the shipped behaviour as broken —
"`stream::resolve_manifest` **rejects** `MANIFEST_NAME_CONVENTION_V2`" — which is
the opposite of what the code does. If you are reading this to find out what the
manifest layer does, read `.claude/rules/manifests.md` first; it is the rule of
record.

## What shipped

`resolve_manifest` **accepts** V2 structural manifests, and V2 is **fail-closed**:
a tensor whose name matches no explicit rule raises
`GrokOzempicError::ManifestV2UnmatchedTensor` instead of falling through to
`defaults`. That is the whole point — routers and norms cannot be silently
ternary-quantized by a name-convention mismatch.

Classification order inside a loaded manifest is
**preserve > fp16 > ternary_candidates > defaults**.

| Input names | Manifest | Behaviour |
|---|---|---|
| structural (`block_{NNN}.slot_{SS}.{kind}`) | `dissect/grok-1/structural-manifest.json` | V2, fail-closed |
| legacy `blk.*` | `dissect/grok-1/baseline.json` | V1, defaults fallthrough allowed |

## Verify it still holds

```bash
# the fail-closed guard, both directions
cargo test --features cli --locked v2_manifest_fails_closed
cargo test --features cli --locked v2_structural_manifest_end_to_end

# the run3 classification oracle (skips when the dissect run is not mounted)
cargo test --features cli --locked run3_conversion_manifest_names_fully_classified
```

⚠ Two known gaps, both tracked:

- The fail-closed guard is **copy-pasted per input format**, and only the npy
  copy is tested — the safetensors path has zero coverage (**GH #103**).
- `GROK_OZEMPIC_DISSECT_RUN` has two incompatible meanings, and the documented
  value makes the run3 oracle **silently skip while passing** (**GH #102**). If
  that test reports `skip:`, believe the skip, not the green.

## Related

- `.claude/rules/manifests.md` — V1 vs V2, classification order, delivery precedence
- `.claude/rules/goz1-pipeline.md` — real pack recipes, and why V2 fail-closed does
  **not** detect under-packing
- `/quantize-embed`, `/pr-ready`
