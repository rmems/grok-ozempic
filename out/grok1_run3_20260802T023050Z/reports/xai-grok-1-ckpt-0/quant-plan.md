# xai-dissect quant plan

- **model_family**: `grok-1`
- **checkpoint**: `/home/raulmc/.models/xai-grok-1/ckpt-0`
- **baseline**: `grok1-map-v1-clean`
- **schema_version**: 1

## Required validation

| Metric | Required | Discovered |
| ------ | -------: | ---------: |
| blocks | 64 | 64 |
| tensors | 770 | 770 |
| routers | 64 | 64 |
| expert_families | 192 | 192 |
| unknown_tensors | 0 | 0 |

## Keep fp32

- `router`
- `block_norm`
- `final_norm`

## Pilot quantize

- `attn_proj_i8.model_width`
- `attn_proj_i8.narrow`
- `moe_expert.gate`
- `moe_expert.up`
- `moe_expert.down`

## Defer

- `token_embedding`

## Notes

- 770 tensors partition into 448 quantization candidates, 64 routing-critical, 257 precision-sensitive, and 1 deferred entries.
- Top quantization candidate is `block_030.slot_01.moe_expert.down` with readiness 0.188.
