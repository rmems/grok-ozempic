# xai-dissect routing report

- **model_family**: `grok-1`
- **checkpoint**: `grok-1-official/ckpt-0`
- **shards**: 770
- **relevant_blocks**: 64
- **expected_experts_per_router**: 8
- **schema_version**: 1

## Candidate routing tensors

| Block | Slot | Shape | Orientation | Experts | Kind | Structural name |
| ----: | ---: | ----- | ----------- | ------: | ---- | --------------- |
| 0 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_000.routing_slot_11` |
| 1 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_001.routing_slot_11` |
| 2 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_002.routing_slot_11` |
| 3 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_003.routing_slot_11` |
| 4 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_004.routing_slot_11` |
| 5 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_005.routing_slot_11` |
| 6 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_006.routing_slot_11` |
| 7 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_007.routing_slot_11` |
| 8 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_008.routing_slot_11` |
| 9 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_009.routing_slot_11` |
| 10 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_010.routing_slot_11` |
| 11 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_011.routing_slot_11` |
| 12 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_012.routing_slot_11` |
| 13 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_013.routing_slot_11` |
| 14 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_014.routing_slot_11` |
| 15 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_015.routing_slot_11` |
| 16 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_016.routing_slot_11` |
| 17 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_017.routing_slot_11` |
| 18 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_018.routing_slot_11` |
| 19 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_019.routing_slot_11` |
| 20 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_020.routing_slot_11` |
| 21 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_021.routing_slot_11` |
| 22 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_022.routing_slot_11` |
| 23 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_023.routing_slot_11` |
| 24 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_024.routing_slot_11` |
| 25 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_025.routing_slot_11` |
| 26 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_026.routing_slot_11` |
| 27 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_027.routing_slot_11` |
| 28 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_028.routing_slot_11` |
| 29 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_029.routing_slot_11` |
| 30 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_030.routing_slot_11` |
| 31 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_031.routing_slot_11` |
| 32 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_032.routing_slot_11` |
| 33 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_033.routing_slot_11` |
| 34 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_034.routing_slot_11` |
| 35 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_035.routing_slot_11` |
| 36 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_036.routing_slot_11` |
| 37 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_037.routing_slot_11` |
| 38 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_038.routing_slot_11` |
| 39 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_039.routing_slot_11` |
| 40 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_040.routing_slot_11` |
| 41 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_041.routing_slot_11` |
| 42 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_042.routing_slot_11` |
| 43 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_043.routing_slot_11` |
| 44 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_044.routing_slot_11` |
| 45 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_045.routing_slot_11` |
| 46 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_046.routing_slot_11` |
| 47 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_047.routing_slot_11` |
| 48 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_048.routing_slot_11` |
| 49 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_049.routing_slot_11` |
| 50 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_050.routing_slot_11` |
| 51 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_051.routing_slot_11` |
| 52 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_052.routing_slot_11` |
| 53 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_053.routing_slot_11` |
| 54 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_054.routing_slot_11` |
| 55 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_055.routing_slot_11` |
| 56 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_056.routing_slot_11` |
| 57 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_057.routing_slot_11` |
| 58 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_058.routing_slot_11` |
| 59 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_059.routing_slot_11` |
| 60 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_060.routing_slot_11` |
| 61 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_061.routing_slot_11` |
| 62 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_062.routing_slot_11` |
| 63 | 11 | `(6144, 8)` | d_model_to_experts | 8 | router | `block_063.routing_slot_11` |
