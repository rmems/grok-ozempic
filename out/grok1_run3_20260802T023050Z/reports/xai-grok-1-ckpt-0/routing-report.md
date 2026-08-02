# xai-dissect routing report

- **model_family**: `grok-1`
- **checkpoint**: `/home/raulmc/.models/xai-grok-1/ckpt-0`
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

## Shape and orientation summaries

| Orientation | Count | Blocks | Shapes |
| ----------- | ----: | -----: | ------ |
| d_model_to_experts | 64 | 64 | (6144, 8) |

## Layer-by-layer routing metadata

| Label | Block | Local experts | Primary candidate | Candidate count |
| ----- | ----: | ------------: | ----------------- | --------------: |
| block_000 | 0 | 8 | shard 13 idx 0 slot 11 | 1 |
| block_001 | 1 | 8 | shard 25 idx 0 slot 11 | 1 |
| block_002 | 2 | 8 | shard 37 idx 0 slot 11 | 1 |
| block_003 | 3 | 8 | shard 49 idx 0 slot 11 | 1 |
| block_004 | 4 | 8 | shard 61 idx 0 slot 11 | 1 |
| block_005 | 5 | 8 | shard 73 idx 0 slot 11 | 1 |
| block_006 | 6 | 8 | shard 85 idx 0 slot 11 | 1 |
| block_007 | 7 | 8 | shard 97 idx 0 slot 11 | 1 |
| block_008 | 8 | 8 | shard 109 idx 0 slot 11 | 1 |
| block_009 | 9 | 8 | shard 121 idx 0 slot 11 | 1 |
| block_010 | 10 | 8 | shard 133 idx 0 slot 11 | 1 |
| block_011 | 11 | 8 | shard 145 idx 0 slot 11 | 1 |
| block_012 | 12 | 8 | shard 157 idx 0 slot 11 | 1 |
| block_013 | 13 | 8 | shard 169 idx 0 slot 11 | 1 |
| block_014 | 14 | 8 | shard 181 idx 0 slot 11 | 1 |
| block_015 | 15 | 8 | shard 193 idx 0 slot 11 | 1 |
| block_016 | 16 | 8 | shard 205 idx 0 slot 11 | 1 |
| block_017 | 17 | 8 | shard 217 idx 0 slot 11 | 1 |
| block_018 | 18 | 8 | shard 229 idx 0 slot 11 | 1 |
| block_019 | 19 | 8 | shard 241 idx 0 slot 11 | 1 |
| block_020 | 20 | 8 | shard 253 idx 0 slot 11 | 1 |
| block_021 | 21 | 8 | shard 265 idx 0 slot 11 | 1 |
| block_022 | 22 | 8 | shard 277 idx 0 slot 11 | 1 |
| block_023 | 23 | 8 | shard 289 idx 0 slot 11 | 1 |
| block_024 | 24 | 8 | shard 301 idx 0 slot 11 | 1 |
| block_025 | 25 | 8 | shard 313 idx 0 slot 11 | 1 |
| block_026 | 26 | 8 | shard 325 idx 0 slot 11 | 1 |
| block_027 | 27 | 8 | shard 337 idx 0 slot 11 | 1 |
| block_028 | 28 | 8 | shard 349 idx 0 slot 11 | 1 |
| block_029 | 29 | 8 | shard 361 idx 0 slot 11 | 1 |
| block_030 | 30 | 8 | shard 373 idx 0 slot 11 | 1 |
| block_031 | 31 | 8 | shard 385 idx 0 slot 11 | 1 |
| block_032 | 32 | 8 | shard 397 idx 0 slot 11 | 1 |
| block_033 | 33 | 8 | shard 409 idx 0 slot 11 | 1 |
| block_034 | 34 | 8 | shard 421 idx 0 slot 11 | 1 |
| block_035 | 35 | 8 | shard 433 idx 0 slot 11 | 1 |
| block_036 | 36 | 8 | shard 445 idx 0 slot 11 | 1 |
| block_037 | 37 | 8 | shard 457 idx 0 slot 11 | 1 |
| block_038 | 38 | 8 | shard 469 idx 0 slot 11 | 1 |
| block_039 | 39 | 8 | shard 481 idx 0 slot 11 | 1 |
| block_040 | 40 | 8 | shard 493 idx 0 slot 11 | 1 |
| block_041 | 41 | 8 | shard 505 idx 0 slot 11 | 1 |
| block_042 | 42 | 8 | shard 517 idx 0 slot 11 | 1 |
| block_043 | 43 | 8 | shard 529 idx 0 slot 11 | 1 |
| block_044 | 44 | 8 | shard 541 idx 0 slot 11 | 1 |
| block_045 | 45 | 8 | shard 553 idx 0 slot 11 | 1 |
| block_046 | 46 | 8 | shard 565 idx 0 slot 11 | 1 |
| block_047 | 47 | 8 | shard 577 idx 0 slot 11 | 1 |
| block_048 | 48 | 8 | shard 589 idx 0 slot 11 | 1 |
| block_049 | 49 | 8 | shard 601 idx 0 slot 11 | 1 |
| block_050 | 50 | 8 | shard 613 idx 0 slot 11 | 1 |
| block_051 | 51 | 8 | shard 625 idx 0 slot 11 | 1 |
| block_052 | 52 | 8 | shard 637 idx 0 slot 11 | 1 |
| block_053 | 53 | 8 | shard 649 idx 0 slot 11 | 1 |
| block_054 | 54 | 8 | shard 661 idx 0 slot 11 | 1 |
| block_055 | 55 | 8 | shard 673 idx 0 slot 11 | 1 |
| block_056 | 56 | 8 | shard 685 idx 0 slot 11 | 1 |
| block_057 | 57 | 8 | shard 697 idx 0 slot 11 | 1 |
| block_058 | 58 | 8 | shard 709 idx 0 slot 11 | 1 |
| block_059 | 59 | 8 | shard 721 idx 0 slot 11 | 1 |
| block_060 | 60 | 8 | shard 733 idx 0 slot 11 | 1 |
| block_061 | 61 | 8 | shard 745 idx 0 slot 11 | 1 |
| block_062 | 62 | 8 | shard 757 idx 0 slot 11 | 1 |
| block_063 | 63 | 8 | shard 769 idx 0 slot 11 | 1 |

## Gate tensor structural metrics

| Structural name | Input width | Output width | Logits/input | Bytes |
| --------------- | ----------: | -----------: | -----------: | ----: |
| `block_000.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_001.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_002.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_003.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_004.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_005.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_006.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_007.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_008.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_009.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_010.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_011.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_012.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_013.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_014.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_015.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_016.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_017.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_018.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_019.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_020.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_021.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_022.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_023.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_024.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_025.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_026.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_027.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_028.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_029.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_030.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_031.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_032.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_033.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_034.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_035.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_036.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_037.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_038.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_039.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_040.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_041.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_042.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_043.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_044.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_045.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_046.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_047.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_048.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_049.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_050.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_051.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_052.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_053.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_054.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_055.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_056.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_057.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_058.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_059.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_060.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_061.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_062.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |
| `block_063.routing_slot_11` | 6144 | 8 | 8 | 196608 (192.00 KiB) |

## Expert count linkage

| Structural name | Linked experts | Matches inferred experts |
| --------------- | -------------: | ----------------------- |
| `block_000.routing_slot_11` | 8 | yes |
| `block_001.routing_slot_11` | 8 | yes |
| `block_002.routing_slot_11` | 8 | yes |
| `block_003.routing_slot_11` | 8 | yes |
| `block_004.routing_slot_11` | 8 | yes |
| `block_005.routing_slot_11` | 8 | yes |
| `block_006.routing_slot_11` | 8 | yes |
| `block_007.routing_slot_11` | 8 | yes |
| `block_008.routing_slot_11` | 8 | yes |
| `block_009.routing_slot_11` | 8 | yes |
| `block_010.routing_slot_11` | 8 | yes |
| `block_011.routing_slot_11` | 8 | yes |
| `block_012.routing_slot_11` | 8 | yes |
| `block_013.routing_slot_11` | 8 | yes |
| `block_014.routing_slot_11` | 8 | yes |
| `block_015.routing_slot_11` | 8 | yes |
| `block_016.routing_slot_11` | 8 | yes |
| `block_017.routing_slot_11` | 8 | yes |
| `block_018.routing_slot_11` | 8 | yes |
| `block_019.routing_slot_11` | 8 | yes |
| `block_020.routing_slot_11` | 8 | yes |
| `block_021.routing_slot_11` | 8 | yes |
| `block_022.routing_slot_11` | 8 | yes |
| `block_023.routing_slot_11` | 8 | yes |
| `block_024.routing_slot_11` | 8 | yes |
| `block_025.routing_slot_11` | 8 | yes |
| `block_026.routing_slot_11` | 8 | yes |
| `block_027.routing_slot_11` | 8 | yes |
| `block_028.routing_slot_11` | 8 | yes |
| `block_029.routing_slot_11` | 8 | yes |
| `block_030.routing_slot_11` | 8 | yes |
| `block_031.routing_slot_11` | 8 | yes |
| `block_032.routing_slot_11` | 8 | yes |
| `block_033.routing_slot_11` | 8 | yes |
| `block_034.routing_slot_11` | 8 | yes |
| `block_035.routing_slot_11` | 8 | yes |
| `block_036.routing_slot_11` | 8 | yes |
| `block_037.routing_slot_11` | 8 | yes |
| `block_038.routing_slot_11` | 8 | yes |
| `block_039.routing_slot_11` | 8 | yes |
| `block_040.routing_slot_11` | 8 | yes |
| `block_041.routing_slot_11` | 8 | yes |
| `block_042.routing_slot_11` | 8 | yes |
| `block_043.routing_slot_11` | 8 | yes |
| `block_044.routing_slot_11` | 8 | yes |
| `block_045.routing_slot_11` | 8 | yes |
| `block_046.routing_slot_11` | 8 | yes |
| `block_047.routing_slot_11` | 8 | yes |
| `block_048.routing_slot_11` | 8 | yes |
| `block_049.routing_slot_11` | 8 | yes |
| `block_050.routing_slot_11` | 8 | yes |
| `block_051.routing_slot_11` | 8 | yes |
| `block_052.routing_slot_11` | 8 | yes |
| `block_053.routing_slot_11` | 8 | yes |
| `block_054.routing_slot_11` | 8 | yes |
| `block_055.routing_slot_11` | 8 | yes |
| `block_056.routing_slot_11` | 8 | yes |
| `block_057.routing_slot_11` | 8 | yes |
| `block_058.routing_slot_11` | 8 | yes |
| `block_059.routing_slot_11` | 8 | yes |
| `block_060.routing_slot_11` | 8 | yes |
| `block_061.routing_slot_11` | 8 | yes |
| `block_062.routing_slot_11` | 8 | yes |
| `block_063.routing_slot_11` | 8 | yes |

## Likely routing-critical blocks

| Block | Label | Reason |
| ----: | ----- | ------ |
| 0 | block_000 | contains a primary routing candidate linked to a 8-expert MoE block |
| 1 | block_001 | contains a primary routing candidate linked to a 8-expert MoE block |
| 2 | block_002 | contains a primary routing candidate linked to a 8-expert MoE block |
| 3 | block_003 | contains a primary routing candidate linked to a 8-expert MoE block |
| 4 | block_004 | contains a primary routing candidate linked to a 8-expert MoE block |
| 5 | block_005 | contains a primary routing candidate linked to a 8-expert MoE block |
| 6 | block_006 | contains a primary routing candidate linked to a 8-expert MoE block |
| 7 | block_007 | contains a primary routing candidate linked to a 8-expert MoE block |
| 8 | block_008 | contains a primary routing candidate linked to a 8-expert MoE block |
| 9 | block_009 | contains a primary routing candidate linked to a 8-expert MoE block |
| 10 | block_010 | contains a primary routing candidate linked to a 8-expert MoE block |
| 11 | block_011 | contains a primary routing candidate linked to a 8-expert MoE block |
| 12 | block_012 | contains a primary routing candidate linked to a 8-expert MoE block |
| 13 | block_013 | contains a primary routing candidate linked to a 8-expert MoE block |
| 14 | block_014 | contains a primary routing candidate linked to a 8-expert MoE block |
| 15 | block_015 | contains a primary routing candidate linked to a 8-expert MoE block |
| 16 | block_016 | contains a primary routing candidate linked to a 8-expert MoE block |
| 17 | block_017 | contains a primary routing candidate linked to a 8-expert MoE block |
| 18 | block_018 | contains a primary routing candidate linked to a 8-expert MoE block |
| 19 | block_019 | contains a primary routing candidate linked to a 8-expert MoE block |
| 20 | block_020 | contains a primary routing candidate linked to a 8-expert MoE block |
| 21 | block_021 | contains a primary routing candidate linked to a 8-expert MoE block |
| 22 | block_022 | contains a primary routing candidate linked to a 8-expert MoE block |
| 23 | block_023 | contains a primary routing candidate linked to a 8-expert MoE block |
| 24 | block_024 | contains a primary routing candidate linked to a 8-expert MoE block |
| 25 | block_025 | contains a primary routing candidate linked to a 8-expert MoE block |
| 26 | block_026 | contains a primary routing candidate linked to a 8-expert MoE block |
| 27 | block_027 | contains a primary routing candidate linked to a 8-expert MoE block |
| 28 | block_028 | contains a primary routing candidate linked to a 8-expert MoE block |
| 29 | block_029 | contains a primary routing candidate linked to a 8-expert MoE block |
| 30 | block_030 | contains a primary routing candidate linked to a 8-expert MoE block |
| 31 | block_031 | contains a primary routing candidate linked to a 8-expert MoE block |
| 32 | block_032 | contains a primary routing candidate linked to a 8-expert MoE block |
| 33 | block_033 | contains a primary routing candidate linked to a 8-expert MoE block |
| 34 | block_034 | contains a primary routing candidate linked to a 8-expert MoE block |
| 35 | block_035 | contains a primary routing candidate linked to a 8-expert MoE block |
| 36 | block_036 | contains a primary routing candidate linked to a 8-expert MoE block |
| 37 | block_037 | contains a primary routing candidate linked to a 8-expert MoE block |
| 38 | block_038 | contains a primary routing candidate linked to a 8-expert MoE block |
| 39 | block_039 | contains a primary routing candidate linked to a 8-expert MoE block |
| 40 | block_040 | contains a primary routing candidate linked to a 8-expert MoE block |
| 41 | block_041 | contains a primary routing candidate linked to a 8-expert MoE block |
| 42 | block_042 | contains a primary routing candidate linked to a 8-expert MoE block |
| 43 | block_043 | contains a primary routing candidate linked to a 8-expert MoE block |
| 44 | block_044 | contains a primary routing candidate linked to a 8-expert MoE block |
| 45 | block_045 | contains a primary routing candidate linked to a 8-expert MoE block |
| 46 | block_046 | contains a primary routing candidate linked to a 8-expert MoE block |
| 47 | block_047 | contains a primary routing candidate linked to a 8-expert MoE block |
| 48 | block_048 | contains a primary routing candidate linked to a 8-expert MoE block |
| 49 | block_049 | contains a primary routing candidate linked to a 8-expert MoE block |
| 50 | block_050 | contains a primary routing candidate linked to a 8-expert MoE block |
| 51 | block_051 | contains a primary routing candidate linked to a 8-expert MoE block |
| 52 | block_052 | contains a primary routing candidate linked to a 8-expert MoE block |
| 53 | block_053 | contains a primary routing candidate linked to a 8-expert MoE block |
| 54 | block_054 | contains a primary routing candidate linked to a 8-expert MoE block |
| 55 | block_055 | contains a primary routing candidate linked to a 8-expert MoE block |
| 56 | block_056 | contains a primary routing candidate linked to a 8-expert MoE block |
| 57 | block_057 | contains a primary routing candidate linked to a 8-expert MoE block |
| 58 | block_058 | contains a primary routing candidate linked to a 8-expert MoE block |
| 59 | block_059 | contains a primary routing candidate linked to a 8-expert MoE block |
| 60 | block_060 | contains a primary routing candidate linked to a 8-expert MoE block |
| 61 | block_061 | contains a primary routing candidate linked to a 8-expert MoE block |
| 62 | block_062 | contains a primary routing candidate linked to a 8-expert MoE block |
| 63 | block_063 | contains a primary routing candidate linked to a 8-expert MoE block |

## Grok-specific layout notes

- Primary routing candidates are plain f32 tensors oriented from d_model to expert logits.
- Primary routing candidates occupy a stable block slot (11) across observed blocks.
- Observed primary routing tensors match the Grok-style router shape `(6144, 8)`.

## Routing anomalies

None detected.

## Missing routing candidates

None detected.
