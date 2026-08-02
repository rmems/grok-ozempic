# xai-dissect Grok-1 pilot selection plan

- **model_family**: `grok-1`
- **checkpoint**: `/home/raulmc/.models/xai-grok-1/ckpt-0`
- **baseline**: `grok1-map-v1-clean`
- **schema_version**: 1

## Selected blocks

| Block | Label | Rationale |
| ----: | ----- | --------- |
| 0 | `block_000` | early baseline |
| 8 | `block_008` | near-zero-sensitive router |
| 28 | `block_028` | near-zero-sensitive router |
| 60 | `block_060` | high readiness/routing-critical sample |
| 63 | `block_063` | late-layer / high peak-to-rms router region |

## Modes

- `attention_only`
- `expert_only`
- `attention_plus_expert`

## Protection rules

- router tensors must remain untouched in every first-pass pilot
- block_norm and final_norm tensors must remain untouched in every first-pass pilot
- pilot artifacts must be emitted per mode and remain comparable across selected blocks

## Expected comparison artifacts

- `pilot-selection-plan.json`
- `pilot-selection-plan.md`
- `route-preservation-report.json`
- `route-preservation-report.md`

## Notes

- This is a planning artifact only; xai-dissect does not mutate checkpoints or execute a quantization runtime.
- Use the selected blocks and protected-family rules to drive downstream bounded pilot runs.
