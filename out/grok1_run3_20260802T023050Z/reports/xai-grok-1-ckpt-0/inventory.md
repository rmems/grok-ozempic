# xai-dissect inventory

- **model_family**: `grok-1`
- **checkpoint**: `/home/raulmc/.models/xai-grok-1/ckpt-0`
- **shards**: 770
- **schema_version**: 2

## Inferred hyperparameters

| Field | Value |
| ----- | ----- |
| vocab_size | 131072 |
| d_model | 6144 |
| n_experts | 8 |
| d_ff | 32768 |
| n_blocks | 64 |

## Totals

| Metric | Value |
| ------ | ----- |
| tensors | 770 |
| f32 tensors | 322 |
| int8 tensors | 448 |
| quant tensors | 448 |
| total elements | 315684820992 |
| total bytes | 318114914304 (296.27 GiB) |

## Tensor kinds

| Kind | Count | Bytes |
| ---- | ----: | ----: |
| attn_proj_i8.model_width | 128 | 4831838208 (4.50 GiB) |
| attn_proj_i8.narrow | 128 | 805306368 (768.00 MiB) |
| block_norm | 256 | 6291456 (6.00 MiB) |
| final_norm | 1 | 24576 (24.00 KiB) |
| moe_expert.down | 64 | 103079215104 (96.00 GiB) |
| moe_expert.gate | 64 | 103079215104 (96.00 GiB) |
| moe_expert.up | 64 | 103079215104 (96.00 GiB) |
| router | 64 | 12582912 (12.00 MiB) |
| token_embedding | 1 | 3221225472 (3.00 GiB) |

## Blocks

| Label | Block | Shards | Tensors | Bytes | Kinds |
| ----- | ----: | ------ | ------: | ----: | ----- |
| embedding | - | 0..=0 | 1 | 3221225472 (3.00 GiB) | 1xtoken_embedding |
| block_000 | 0 | 2..=13 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_001 | 1 | 14..=25 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_002 | 2 | 26..=37 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_003 | 3 | 38..=49 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_004 | 4 | 50..=61 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_005 | 5 | 62..=73 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_006 | 6 | 74..=85 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_007 | 7 | 86..=97 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_008 | 8 | 98..=109 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_009 | 9 | 110..=121 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_010 | 10 | 122..=133 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_011 | 11 | 134..=145 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_012 | 12 | 146..=157 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_013 | 13 | 158..=169 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_014 | 14 | 170..=181 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_015 | 15 | 182..=193 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_016 | 16 | 194..=205 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_017 | 17 | 206..=217 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_018 | 18 | 218..=229 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_019 | 19 | 230..=241 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_020 | 20 | 242..=253 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_021 | 21 | 254..=265 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_022 | 22 | 266..=277 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_023 | 23 | 278..=289 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_024 | 24 | 290..=301 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_025 | 25 | 302..=313 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_026 | 26 | 314..=325 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_027 | 27 | 326..=337 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_028 | 28 | 338..=349 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_029 | 29 | 350..=361 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_030 | 30 | 362..=373 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_031 | 31 | 374..=385 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_032 | 32 | 386..=397 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_033 | 33 | 398..=409 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_034 | 34 | 410..=421 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_035 | 35 | 422..=433 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_036 | 36 | 434..=445 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_037 | 37 | 446..=457 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_038 | 38 | 458..=469 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_039 | 39 | 470..=481 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_040 | 40 | 482..=493 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_041 | 41 | 494..=505 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_042 | 42 | 506..=517 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_043 | 43 | 518..=529 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_044 | 44 | 530..=541 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_045 | 45 | 542..=553 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_046 | 46 | 554..=565 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_047 | 47 | 566..=577 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_048 | 48 | 578..=589 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_049 | 49 | 590..=601 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_050 | 50 | 602..=613 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_051 | 51 | 614..=625 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_052 | 52 | 626..=637 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_053 | 53 | 638..=649 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_054 | 54 | 650..=661 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_055 | 55 | 662..=673 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_056 | 56 | 674..=685 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_057 | 57 | 686..=697 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_058 | 58 | 698..=709 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_059 | 59 | 710..=721 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_060 | 60 | 722..=733 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_061 | 61 | 734..=745 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_062 | 62 | 746..=757 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| block_063 | 63 | 758..=769 | 12 | 4920213504 (4.58 GiB) | 2xattn_proj_i8.model_width, 2xattn_proj_i8.narrow, 4xblock_norm, 1xmoe_expert.down, 1xmoe_expert.gate, 1xmoe_expert.up, 1xrouter |
| final_norm | - | 1..=1 | 1 | 24576 (24.00 KiB) | 1xfinal_norm |

## Exemplar block (`block_000`)

| Shard | In-shard | Role | Dtype | Shape | Kind | Slot |
| ----: | -------: | ---- | ----- | ----- | ---- | ---: |
| 2 | 0 | quant.weight | int8 | `(8, 6144, 32768)` | moe_expert.gate | 0 |
| 3 | 0 | quant.weight | int8 | `(8, 32768, 6144)` | moe_expert.down | 1 |
| 4 | 0 | quant.weight | int8 | `(8, 6144, 32768)` | moe_expert.up | 2 |
| 5 | 0 | quant.weight | int8 | `(6144, 1024)` | attn_proj_i8.narrow | 3 |
| 6 | 0 | quant.weight | int8 | `(6144, 6144)` | attn_proj_i8.model_width | 4 |
| 7 | 0 | quant.weight | int8 | `(6144, 6144)` | attn_proj_i8.model_width | 5 |
| 8 | 0 | quant.weight | int8 | `(6144, 1024)` | attn_proj_i8.narrow | 6 |
| 9 | 0 | tensor | f32 | `(6144,)` | block_norm | 7 |
| 10 | 0 | tensor | f32 | `(6144,)` | block_norm | 8 |
| 11 | 0 | tensor | f32 | `(6144,)` | block_norm | 9 |
| 12 | 0 | tensor | f32 | `(6144,)` | block_norm | 10 |
| 13 | 0 | tensor | f32 | `(6144, 8)` | router | 11 |
