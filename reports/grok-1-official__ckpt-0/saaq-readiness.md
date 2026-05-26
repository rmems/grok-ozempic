# xai-dissect SAAQ-readiness report

- **model_family**: `grok-1`
- **checkpoint**: `grok-1-official/ckpt-0`
- **shards**: 770
- **candidate_targets**: 1
- **routing_critical_tensors**: 64
- **schema_version**: 1

## Candidate target tensors

| Rank | Tensor | Kind | Region | Readiness | Opportunity | Risk | Disposition |
| ---: | ------ | ---- | ------ | --------: | ----------: | ---: | ----------- |
| 1 | `embedding.slot_00.token_embedding` | token_embedding | embedding_heavy | 0.176 | 0.331 | 0.391 | candidate |

## Routing-critical tensors

| Tensor | Readiness | Risk | Reasons |
| ------ | --------: | ---: | ------- |
| `block_000.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_001.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_002.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_003.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_004.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_005.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_006.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_007.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_008.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_009.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_010.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_011.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_012.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_013.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_014.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_015.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_016.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_017.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_018.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_019.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_020.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_021.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_022.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_023.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_024.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_025.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_026.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_027.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_028.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_029.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_030.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_031.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_032.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_033.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_034.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_035.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_036.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_037.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_038.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_039.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_040.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_041.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_042.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_043.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_044.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_045.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_046.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_047.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_048.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_049.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_050.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_051.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_052.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_053.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_054.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_055.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_056.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_057.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_058.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_059.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_060.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_061.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_062.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_063.slot_11.router` | 0.054 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
