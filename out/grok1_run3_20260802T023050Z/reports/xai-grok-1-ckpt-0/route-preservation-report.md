# xai-dissect Grok-1 route-preservation report

- **model_family**: `grok-1`
- **checkpoint**: `/home/raulmc/.models/xai-grok-1/ckpt-0`
- **baseline**: `grok1-map-v1-clean`
- **schema_version**: 1


## Router metrics

| Metric | Scope | Status | Threshold | Observed | Detail |
| ------ | ----- | ------ | --------- | -------- | ------ |
| `router_top1_agreement` | router_behavior | unknown | >= 99.0% | - | Threshold reserved for downstream pilot comparison artifacts; xai-dissect defines the gate but does not execute pilot inference. |
| `router_top2_set_agreement` | router_behavior | unknown | >= 99.5% | - | Threshold reserved for downstream pilot comparison artifacts; report as first-class sprint evidence when available. |
| `expert_load_distribution_delta` | router_behavior | unknown | - | - | Capture expert-load distribution drift once bounded pilot routing traces are available. |
| `expert_load_js_divergence` | router_behavior | unknown | - | - | Report JS/KL-style divergence over expert-load distributions when downstream pilot evidence exists. |
| `router_logit_rank_correlation` | router_behavior | unknown | - | - | Report rank correlation for router logits when logits are captured by downstream pilot comparisons. |

## Block metrics

| Metric | Scope | Status | Threshold | Observed | Detail |
| ------ | ----- | ------ | --------- | -------- | ------ |
| `block_output_cosine` | block_behavior | unknown | >= 0.995 | - | Tracked as a go/no-go threshold once bounded pilot outputs exist. |
| `block_output_rmse` | block_behavior | unknown | - | - | Report alongside cosine similarity for bounded pilot comparisons. |
| `residual_stream_drift` | block_behavior | unknown | - | - | Summarize residual-stream drift once downstream pilot artifacts provide comparable block activations. |

## Weight metrics

| Metric | Scope | Status | Threshold | Observed | Detail |
| ------ | ----- | ------ | --------- | -------- | ------ |
| `weight_reconstruction_mse` | weight_reconstruction | unknown | - | - | Generic reconstruction metrics remain secondary to router-behavior preservation for Grok-1 MoE validation. |
| `weight_cosine_similarity` | weight_reconstruction | unknown | - | - | Useful companion metric, but not sufficient by itself to clear a full quantization run. |
| `weight_max_absolute_error` | weight_reconstruction | unknown | - | - | Report max absolute reconstruction error when downstream pilot comparisons include raw tensor deltas. |
| `per_channel_scale_error_summary` | weight_reconstruction | unknown | - | - | Summarize per-channel scale/error drift where quantization metadata is available. |

## Model metrics

| Metric | Scope | Status | Threshold | Observed | Detail |
| ------ | ----- | ------ | --------- | -------- | ------ |
| `logit_kl` | model_behavior | unknown | - | - | Report-only placeholder for model/logit KL when downstream pilot inference captures logits. |
| `perplexity_delta` | model_behavior | unknown | - | - | Report-only placeholder for calibration-data perplexity delta once downstream pilot evaluation exists. |
| `generation_sanity_summary` | model_behavior | unknown | - | - | Report-only placeholder for short generation sanity checks when pilot inference is available. |

## Notes

- This report defines the required route-preservation surface and thresholds for Grok-1 pilot evidence.
- Statuses remain unknown until downstream pilot artifacts supply the observed values.
