# xai-dissect stats report

- **model_family**: `grok-1`
- **checkpoint**: `/home/raulmc/.models/xai-grok-1/ckpt-0`
- **shards**: 770
- **sample_values_per_tensor**: 65536
- **schema_version**: 1

## Norm summary

- **mean_rms**: 19.762282

### Top RMS tensors

| Tensor | Kind | Block | Value |
| ------ | ---- | ----: | ----: |
| `block_063.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 63 | 37.729480 |
| `block_022.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 22 | 37.665003 |
| `block_026.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 26 | 37.589824 |
| `block_023.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 23 | 37.552870 |
| `block_045.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 45 | 37.542414 |
| `block_049.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 49 | 37.540125 |
| `block_034.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 34 | 37.533975 |
| `block_027.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 27 | 37.507881 |
| `block_012.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 12 | 37.480414 |
| `block_046.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 46 | 37.466894 |

### Top L2 tensors

| Tensor | Kind | Block | Value |
| ------ | ---- | ----: | ----: |
| `block_063.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 63 | 9658.746813 |
| `block_022.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 22 | 9642.240715 |
| `block_026.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 26 | 9622.994908 |
| `block_023.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 23 | 9613.534678 |
| `block_045.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 45 | 9610.858078 |
| `block_049.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 49 | 9610.272109 |
| `block_034.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 34 | 9608.697518 |
| `block_027.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 27 | 9602.017653 |
| `block_012.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 12 | 9594.986034 |
| `block_046.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 46 | 9591.524853 |

## Variance summary

- **mean_variance**: 631.450634

### Top variance tensors

| Tensor | Kind | Block | Value |
| ------ | ---- | ----: | ----: |
| `block_063.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 63 | 1423.501918 |
| `block_022.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 22 | 1418.610278 |
| `block_026.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 26 | 1412.993531 |
| `block_023.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 23 | 1410.217859 |
| `block_045.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 45 | 1409.430221 |
| `block_049.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 49 | 1409.258567 |
| `block_034.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 34 | 1408.547543 |
| `block_027.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 27 | 1406.821366 |
| `block_012.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 12 | 1404.771217 |
| `block_046.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | 46 | 1403.706608 |

### Lowest variance tensors

| Tensor | Kind | Block | Value |
| ------ | ---- | ----: | ----: |
| `block_028.slot_11.router` | router | 28 | 0.000011 |
| `block_008.slot_11.router` | router | 8 | 0.000011 |
| `block_038.slot_11.router` | router | 38 | 0.000024 |
| `block_006.slot_11.router` | router | 6 | 0.000028 |
| `block_007.slot_11.router` | router | 7 | 0.000035 |
| `block_049.slot_11.router` | router | 49 | 0.000042 |
| `block_058.slot_11.router` | router | 58 | 0.000046 |
| `block_017.slot_11.router` | router | 17 | 0.000050 |
| `block_042.slot_11.router` | router | 42 | 0.000056 |
| `block_060.slot_11.router` | router | 60 | 0.000062 |

## Outlier summary

- **mean_outlier_fraction**: 0.001079

### Most outlier-heavy tensors

| Tensor | Kind | Block | Value |
| ------ | ---- | ----: | ----: |
| `block_053.slot_10.block_norm` | block_norm | 53 | 0.009440 |
| `block_052.slot_10.block_norm` | block_norm | 52 | 0.008626 |
| `block_054.slot_10.block_norm` | block_norm | 54 | 0.008626 |
| `block_025.slot_09.block_norm` | block_norm | 25 | 0.008138 |
| `block_032.slot_07.block_norm` | block_norm | 32 | 0.008138 |
| `block_000.slot_10.block_norm` | block_norm | 0 | 0.007975 |
| `block_055.slot_10.block_norm` | block_norm | 55 | 0.007975 |
| `block_020.slot_09.block_norm` | block_norm | 20 | 0.007812 |
| `block_024.slot_09.block_norm` | block_norm | 24 | 0.007812 |
| `block_047.slot_07.block_norm` | block_norm | 47 | 0.007812 |

### Highest peak-to-RMS tensors

| Tensor | Kind | Block | Value |
| ------ | ---- | ----: | ----: |
| `block_063.slot_11.router` | router | 63 | 15.632256 |
| `block_011.slot_11.router` | router | 11 | 15.616445 |
| `block_033.slot_11.router` | router | 33 | 15.542044 |
| `block_062.slot_08.block_norm` | block_norm | 62 | 15.217553 |
| `block_061.slot_10.block_norm` | block_norm | 61 | 15.030719 |
| `block_062.slot_11.router` | router | 62 | 14.232501 |
| `block_047.slot_11.router` | router | 47 | 13.935002 |
| `block_063.slot_08.block_norm` | block_norm | 63 | 13.648266 |
| `block_063.slot_10.block_norm` | block_norm | 63 | 13.648187 |
| `block_061.slot_08.block_norm` | block_norm | 61 | 13.596289 |

## Per-layer metrics

| Label | Block | Tensors | Bytes | Mean RMS | Mean variance | Mean outlier frac | Routing tensors | Candidate-like tensors |
| ----- | ----: | ------: | ----: | -------: | ------------: | ----------------: | --------------: | ---------------------: |
| embedding | - | 1 | 3221225472 (3.00 GiB) | 0.012307 | 0.000122 | 0.000000 | 0 | 1 |
| final_norm | - | 1 | 24576 (24.00 KiB) | 13.215221 | 1.353698 | 0.002441 | 0 | 0 |
| block_000 | 0 | 12 | 4920213504 (4.58 GiB) | 18.938559 | 604.963556 | 0.002214 | 1 | 0 |
| block_001 | 1 | 12 | 4920213504 (4.58 GiB) | 19.110906 | 618.298176 | 0.000185 | 1 | 0 |
| block_002 | 2 | 12 | 4920213504 (4.58 GiB) | 19.634461 | 646.052209 | 0.000559 | 1 | 0 |
| block_003 | 3 | 12 | 4920213504 (4.58 GiB) | 19.847929 | 627.889121 | 0.000709 | 1 | 0 |
| block_004 | 4 | 12 | 4920213504 (4.58 GiB) | 19.520633 | 628.935815 | 0.000437 | 1 | 0 |
| block_005 | 5 | 12 | 4920213504 (4.58 GiB) | 19.551844 | 622.356439 | 0.000370 | 1 | 0 |
| block_006 | 6 | 12 | 4920213504 (4.58 GiB) | 19.501161 | 618.411681 | 0.000351 | 1 | 0 |
| block_007 | 7 | 12 | 4920213504 (4.58 GiB) | 19.656999 | 619.252349 | 0.000125 | 1 | 0 |
| block_008 | 8 | 12 | 4920213504 (4.58 GiB) | 19.687486 | 626.772627 | 0.000122 | 1 | 0 |
| block_009 | 9 | 12 | 4920213504 (4.58 GiB) | 19.029711 | 597.394976 | 0.000156 | 1 | 0 |
| block_010 | 10 | 12 | 4920213504 (4.58 GiB) | 19.609828 | 613.407568 | 0.000124 | 1 | 0 |
| block_011 | 11 | 12 | 4920213504 (4.58 GiB) | 19.521516 | 626.743719 | 0.000385 | 1 | 0 |
| block_012 | 12 | 12 | 4920213504 (4.58 GiB) | 19.178580 | 624.490430 | 0.000222 | 1 | 0 |
| block_013 | 13 | 12 | 4920213504 (4.58 GiB) | 19.633558 | 632.996656 | 0.000490 | 1 | 0 |
| block_014 | 14 | 12 | 4920213504 (4.58 GiB) | 19.723074 | 631.285295 | 0.000617 | 1 | 0 |
| block_015 | 15 | 12 | 4920213504 (4.58 GiB) | 19.597350 | 625.613969 | 0.000909 | 1 | 0 |
| block_016 | 16 | 12 | 4920213504 (4.58 GiB) | 20.059661 | 639.591558 | 0.001153 | 1 | 0 |
| block_017 | 17 | 12 | 4920213504 (4.58 GiB) | 19.762744 | 630.829732 | 0.001144 | 1 | 0 |
| block_018 | 18 | 12 | 4920213504 (4.58 GiB) | 19.833888 | 641.988214 | 0.001075 | 1 | 0 |
| block_019 | 19 | 12 | 4920213504 (4.58 GiB) | 19.916544 | 636.789853 | 0.001117 | 1 | 0 |
| block_020 | 20 | 12 | 4920213504 (4.58 GiB) | 19.947364 | 646.446714 | 0.001582 | 1 | 0 |
| block_021 | 21 | 12 | 4920213504 (4.58 GiB) | 19.866493 | 637.595649 | 0.001641 | 1 | 0 |
| block_022 | 22 | 12 | 4920213504 (4.58 GiB) | 19.978952 | 650.967210 | 0.001321 | 1 | 0 |
| block_023 | 23 | 12 | 4920213504 (4.58 GiB) | 19.455712 | 630.959436 | 0.000278 | 1 | 0 |
| block_024 | 24 | 12 | 4920213504 (4.58 GiB) | 20.033071 | 645.365361 | 0.001541 | 1 | 0 |
| block_025 | 25 | 12 | 4920213504 (4.58 GiB) | 19.881538 | 641.642472 | 0.001762 | 1 | 0 |
| block_026 | 26 | 12 | 4920213504 (4.58 GiB) | 19.844276 | 642.407453 | 0.001621 | 1 | 0 |
| block_027 | 27 | 12 | 4920213504 (4.58 GiB) | 20.395784 | 648.749726 | 0.001568 | 1 | 0 |
| block_028 | 28 | 12 | 4920213504 (4.58 GiB) | 20.123477 | 638.563007 | 0.001555 | 1 | 0 |
| block_029 | 29 | 12 | 4920213504 (4.58 GiB) | 20.419695 | 644.659119 | 0.001339 | 1 | 0 |
| block_030 | 30 | 12 | 4920213504 (4.58 GiB) | 19.101741 | 597.857240 | 0.001188 | 1 | 0 |
| block_031 | 31 | 12 | 4920213504 (4.58 GiB) | 19.988633 | 651.134375 | 0.001499 | 1 | 0 |
| block_032 | 32 | 12 | 4920213504 (4.58 GiB) | 20.079691 | 654.582224 | 0.001912 | 1 | 0 |
| block_033 | 33 | 12 | 4920213504 (4.58 GiB) | 19.662093 | 626.924492 | 0.001550 | 1 | 0 |
| block_034 | 34 | 12 | 4920213504 (4.58 GiB) | 19.524649 | 644.051058 | 0.000517 | 1 | 0 |
| block_035 | 35 | 12 | 4920213504 (4.58 GiB) | 20.123455 | 654.088716 | 0.001797 | 1 | 0 |
| block_036 | 36 | 12 | 4920213504 (4.58 GiB) | 20.032789 | 646.247397 | 0.001651 | 1 | 0 |
| block_037 | 37 | 12 | 4920213504 (4.58 GiB) | 19.881785 | 642.813031 | 0.001234 | 1 | 0 |
| block_038 | 38 | 12 | 4920213504 (4.58 GiB) | 20.072414 | 639.866064 | 0.001517 | 1 | 0 |
| block_039 | 39 | 12 | 4920213504 (4.58 GiB) | 20.015185 | 634.699947 | 0.001534 | 1 | 0 |
| block_040 | 40 | 12 | 4920213504 (4.58 GiB) | 19.732657 | 633.806914 | 0.001707 | 1 | 0 |
| block_041 | 41 | 12 | 4920213504 (4.58 GiB) | 20.170154 | 650.047077 | 0.001550 | 1 | 0 |
| block_042 | 42 | 12 | 4920213504 (4.58 GiB) | 20.152220 | 641.346227 | 0.001360 | 1 | 0 |
| block_043 | 43 | 12 | 4920213504 (4.58 GiB) | 19.616191 | 621.744772 | 0.001311 | 1 | 0 |
| block_044 | 44 | 12 | 4920213504 (4.58 GiB) | 19.979219 | 633.405641 | 0.001524 | 1 | 0 |
| block_045 | 45 | 12 | 4920213504 (4.58 GiB) | 19.646883 | 642.821325 | 0.000385 | 1 | 0 |
| block_046 | 46 | 12 | 4920213504 (4.58 GiB) | 19.998770 | 636.378549 | 0.001350 | 1 | 0 |
| block_047 | 47 | 12 | 4920213504 (4.58 GiB) | 20.091853 | 639.443239 | 0.001480 | 1 | 0 |
| block_048 | 48 | 12 | 4920213504 (4.58 GiB) | 19.763996 | 628.589709 | 0.001470 | 1 | 0 |
| block_049 | 49 | 12 | 4920213504 (4.58 GiB) | 20.092933 | 642.133333 | 0.001373 | 1 | 0 |
| block_050 | 50 | 12 | 4920213504 (4.58 GiB) | 19.711491 | 623.999660 | 0.001010 | 1 | 0 |
| block_051 | 51 | 12 | 4920213504 (4.58 GiB) | 19.271279 | 594.304210 | 0.001268 | 1 | 0 |
| block_052 | 52 | 12 | 4920213504 (4.58 GiB) | 20.008617 | 638.035032 | 0.001407 | 1 | 0 |
| block_053 | 53 | 12 | 4920213504 (4.58 GiB) | 19.755706 | 617.364183 | 0.001463 | 1 | 0 |
| block_054 | 54 | 12 | 4920213504 (4.58 GiB) | 20.032848 | 628.690998 | 0.001483 | 1 | 0 |
| block_055 | 55 | 12 | 4920213504 (4.58 GiB) | 20.237983 | 642.682205 | 0.001465 | 1 | 0 |
| block_056 | 56 | 12 | 4920213504 (4.58 GiB) | 19.506604 | 640.046216 | 0.000485 | 1 | 0 |
| block_057 | 57 | 12 | 4920213504 (4.58 GiB) | 20.154782 | 627.642276 | 0.001417 | 1 | 0 |
| block_058 | 58 | 12 | 4920213504 (4.58 GiB) | 20.073246 | 612.308275 | 0.001319 | 1 | 0 |
| block_059 | 59 | 12 | 4920213504 (4.58 GiB) | 20.356815 | 626.400690 | 0.001129 | 1 | 0 |
| block_060 | 60 | 12 | 4920213504 (4.58 GiB) | 19.878735 | 615.434489 | 0.000949 | 1 | 0 |
| block_061 | 61 | 12 | 4920213504 (4.58 GiB) | 19.806774 | 659.123538 | 0.000449 | 1 | 0 |
| block_062 | 62 | 12 | 4920213504 (4.58 GiB) | 19.522708 | 641.592163 | 0.000280 | 1 | 0 |
| block_063 | 63 | 12 | 4920213504 (4.58 GiB) | 19.669773 | 646.944190 | 0.000336 | 1 | 0 |

## Per-tensor metrics

| Tensor | Kind | Dtype | Shape | RMS | Variance | Zero frac | Near-zero frac | Outlier frac | Distribution |
| ------ | ---- | ----- | ----- | ---: | -------: | --------: | -------------: | -----------: | ------------ |
| `embedding.slot_00.token_embedding` | token_embedding | f32 | `(131072, 6144)` | 0.012307 | 0.000122 | 0.0000 | 0.0675 | 0.0000 | dense_balanced |
| `final_norm.slot_00.final_norm` | final_norm | f32 | `(6144,)` | 13.215221 | 1.353698 | 0.0000 | 0.0000 | 0.0024 | dense_balanced |
| `block_000.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 27.311491 | 745.902231 | 0.0169 | 0.0500 | 0.0000 | dense_balanced |
| `block_000.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 32.942006 | 1085.169436 | 0.0124 | 0.0371 | 0.0000 | dense_balanced |
| `block_000.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 31.610412 | 999.185917 | 0.0127 | 0.0390 | 0.0000 | dense_balanced |
| `block_000.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.829273 | 1077.752515 | 0.0124 | 0.0373 | 0.0000 | dense_balanced |
| `block_000.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.641375 | 1342.590296 | 0.0112 | 0.0337 | 0.0000 | dense_balanced |
| `block_000.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.644309 | 1001.345806 | 0.0127 | 0.0383 | 0.0000 | dense_balanced |
| `block_000.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.737165 | 1007.247630 | 0.0118 | 0.0377 | 0.0000 | dense_balanced |
| `block_000.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.734543 | 0.126350 | 0.0000 | 0.0000 | 0.0054 | dense_balanced |
| `block_000.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 0.965007 | 0.133454 | 0.0000 | 0.0000 | 0.0068 | dense_balanced |
| `block_000.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.287540 | 0.011616 | 0.0000 | 0.0002 | 0.0063 | dense_balanced |
| `block_000.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 0.536026 | 0.096870 | 0.0000 | 0.0000 | 0.0080 | dense_balanced |
| `block_000.slot_11.router` | router | f32 | `(6144, 8)` | 0.023557 | 0.000555 | 0.0000 | 0.0375 | 0.0000 | dense_balanced |
| `block_001.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.336999 | 1045.678165 | 0.0120 | 0.0361 | 0.0000 | dense_balanced |
| `block_001.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.761474 | 1139.836988 | 0.0123 | 0.0362 | 0.0000 | dense_balanced |
| `block_001.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.938010 | 1084.878956 | 0.0123 | 0.0367 | 0.0000 | dense_balanced |
| `block_001.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.284134 | 917.111061 | 0.0138 | 0.0440 | 0.0000 | dense_balanced |
| `block_001.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.316571 | 1392.496169 | 0.0113 | 0.0334 | 0.0000 | dense_balanced |
| `block_001.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 29.218307 | 853.699918 | 0.0157 | 0.0459 | 0.0000 | dense_balanced |
| `block_001.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.396462 | 985.735044 | 0.0140 | 0.0410 | 0.0000 | dense_balanced |
| `block_001.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.568969 | 0.011858 | 0.0000 | 0.0000 | 0.0005 | dense_balanced |
| `block_001.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 0.565808 | 0.080695 | 0.0000 | 0.0002 | 0.0005 | dense_balanced |
| `block_001.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.401563 | 0.024375 | 0.0000 | 0.0218 | 0.0002 | dense_balanced |
| `block_001.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 0.512656 | 0.023987 | 0.0000 | 0.0000 | 0.0010 | dense_balanced |
| `block_001.slot_11.router` | router | f32 | `(6144, 8)` | 0.029924 | 0.000895 | 0.0000 | 0.0288 | 0.0001 | dense_balanced |
| `block_002.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.729544 | 1071.221255 | 0.0118 | 0.0359 | 0.0000 | dense_balanced |
| `block_002.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.827512 | 1144.267132 | 0.0120 | 0.0356 | 0.0000 | dense_balanced |
| `block_002.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.107933 | 1096.134748 | 0.0124 | 0.0378 | 0.0000 | dense_balanced |
| `block_002.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.005479 | 1024.345451 | 0.0127 | 0.0387 | 0.0000 | dense_balanced |
| `block_002.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.006277 | 1369.368010 | 0.0113 | 0.0352 | 0.0000 | dense_balanced |
| `block_002.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.023238 | 1025.480113 | 0.0127 | 0.0394 | 0.0000 | dense_balanced |
| `block_002.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.963957 | 1021.687552 | 0.0139 | 0.0408 | 0.0000 | dense_balanced |
| `block_002.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.842166 | 0.027352 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_002.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 0.762796 | 0.046183 | 0.0000 | 0.0000 | 0.0055 | dense_balanced |
| `block_002.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.331472 | 0.007221 | 0.0000 | 0.0013 | 0.0000 | dense_balanced |
| `block_002.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 0.980310 | 0.040414 | 0.0000 | 0.0000 | 0.0007 | dense_balanced |
| `block_002.slot_11.router` | router | f32 | `(6144, 8)` | 0.032854 | 0.001079 | 0.0000 | 0.0247 | 0.0005 | dense_balanced |
| `block_003.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.662252 | 1066.813949 | 0.0118 | 0.0368 | 0.0000 | dense_balanced |
| `block_003.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.054134 | 1092.569355 | 0.0125 | 0.0360 | 0.0000 | dense_balanced |
| `block_003.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 31.952342 | 1020.926264 | 0.0127 | 0.0387 | 0.0000 | dense_balanced |
| `block_003.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.205566 | 1037.185912 | 0.0124 | 0.0379 | 0.0000 | dense_balanced |
| `block_003.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.102378 | 1376.523176 | 0.0101 | 0.0329 | 0.0000 | dense_balanced |
| `block_003.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 30.900082 | 954.814900 | 0.0130 | 0.0398 | 0.0000 | dense_balanced |
| `block_003.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.377019 | 984.504450 | 0.0129 | 0.0403 | 0.0000 | dense_balanced |
| `block_003.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 6.364859 | 1.223709 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_003.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 0.903626 | 0.045801 | 0.0000 | 0.0000 | 0.0062 | dense_balanced |
| `block_003.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.483337 | 0.013850 | 0.0000 | 0.0008 | 0.0000 | dense_balanced |
| `block_003.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 1.146756 | 0.047566 | 0.0000 | 0.0000 | 0.0010 | dense_balanced |
| `block_003.slot_11.router` | router | f32 | `(6144, 8)` | 0.022794 | 0.000520 | 0.0000 | 0.0388 | 0.0013 | dense_balanced |
| `block_004.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.593260 | 1062.308280 | 0.0126 | 0.0386 | 0.0000 | dense_balanced |
| `block_004.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.525501 | 1123.958925 | 0.0121 | 0.0369 | 0.0000 | dense_balanced |
| `block_004.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.792551 | 1075.350505 | 0.0122 | 0.0365 | 0.0000 | dense_balanced |
| `block_004.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.374709 | 984.370452 | 0.0136 | 0.0389 | 0.0000 | dense_balanced |
| `block_004.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.816345 | 1355.431003 | 0.0114 | 0.0340 | 0.0000 | dense_balanced |
| `block_004.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.921997 | 1019.008903 | 0.0129 | 0.0379 | 0.0000 | dense_balanced |
| `block_004.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.440793 | 926.590665 | 0.0164 | 0.0475 | 0.0000 | dense_balanced |
| `block_004.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 1.945138 | 0.128307 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_004.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 0.990994 | 0.027573 | 0.0000 | 0.0000 | 0.0036 | dense_balanced |
| `block_004.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.454440 | 0.012466 | 0.0000 | 0.0018 | 0.0000 | dense_balanced |
| `block_004.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 1.367579 | 0.042105 | 0.0000 | 0.0000 | 0.0011 | dense_balanced |
| `block_004.slot_11.router` | router | f32 | `(6144, 8)` | 0.024293 | 0.000590 | 0.0000 | 0.0352 | 0.0005 | dense_balanced |
| `block_005.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.092406 | 1029.888586 | 0.0126 | 0.0381 | 0.0000 | dense_balanced |
| `block_005.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.686484 | 1134.738631 | 0.0115 | 0.0356 | 0.0000 | dense_balanced |
| `block_005.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.390545 | 1049.121329 | 0.0122 | 0.0370 | 0.0000 | dense_balanced |
| `block_005.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.215009 | 974.350886 | 0.0132 | 0.0388 | 0.0000 | dense_balanced |
| `block_005.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.349527 | 1394.987044 | 0.0114 | 0.0339 | 0.0000 | dense_balanced |
| `block_005.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 30.758943 | 946.108734 | 0.0132 | 0.0393 | 0.0000 | dense_balanced |
| `block_005.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.635701 | 938.541698 | 0.0142 | 0.0426 | 0.0000 | dense_balanced |
| `block_005.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 3.646073 | 0.467717 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_005.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 1.068152 | 0.031164 | 0.0000 | 0.0000 | 0.0028 | dense_balanced |
| `block_005.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.285053 | 0.004590 | 0.0000 | 0.0013 | 0.0000 | dense_balanced |
| `block_005.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 1.456256 | 0.035452 | 0.0000 | 0.0000 | 0.0010 | dense_balanced |
| `block_005.slot_11.router` | router | f32 | `(6144, 8)` | 0.037977 | 0.001442 | 0.0000 | 0.0224 | 0.0007 | dense_balanced |
| `block_006.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.708264 | 1069.816648 | 0.0122 | 0.0373 | 0.0000 | dense_balanced |
| `block_006.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.464076 | 1119.771603 | 0.0123 | 0.0366 | 0.0000 | dense_balanced |
| `block_006.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.174225 | 1035.161522 | 0.0125 | 0.0372 | 0.0000 | dense_balanced |
| `block_006.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.992908 | 960.529123 | 0.0135 | 0.0404 | 0.0000 | dense_balanced |
| `block_006.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.251976 | 1314.199544 | 0.0108 | 0.0345 | 0.0000 | dense_balanced |
| `block_006.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.088843 | 1029.669157 | 0.0125 | 0.0386 | 0.0000 | dense_balanced |
| `block_006.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 29.856946 | 891.424184 | 0.0141 | 0.0426 | 0.0000 | dense_balanced |
| `block_006.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 1.798248 | 0.097703 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_006.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 1.256731 | 0.036722 | 0.0000 | 0.0000 | 0.0031 | dense_balanced |
| `block_006.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 1.763132 | 0.162257 | 0.0000 | 0.0011 | 0.0000 | dense_balanced |
| `block_006.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 1.653312 | 0.071687 | 0.0000 | 0.0000 | 0.0008 | dense_balanced |
| `block_006.slot_11.router` | router | f32 | `(6144, 8)` | 0.005271 | 0.000028 | 0.0000 | 0.1552 | 0.0003 | dense_balanced |
| `block_007.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.498279 | 1056.061261 | 0.0120 | 0.0362 | 0.0000 | dense_balanced |
| `block_007.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 32.624095 | 1064.323788 | 0.0120 | 0.0381 | 0.0000 | dense_balanced |
| `block_007.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.700753 | 1069.326176 | 0.0126 | 0.0373 | 0.0000 | dense_balanced |
| `block_007.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.895061 | 954.467280 | 0.0130 | 0.0388 | 0.0000 | dense_balanced |
| `block_007.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.223711 | 1385.581472 | 0.0106 | 0.0336 | 0.0000 | dense_balanced |
| `block_007.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 30.538525 | 932.589858 | 0.0140 | 0.0423 | 0.0000 | dense_balanced |
| `block_007.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.112616 | 967.986936 | 0.0143 | 0.0411 | 0.0000 | dense_balanced |
| `block_007.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 3.862439 | 0.489341 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_007.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 1.261612 | 0.052237 | 0.0000 | 0.0000 | 0.0003 | dense_balanced |
| `block_007.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 1.498972 | 0.096941 | 0.0000 | 0.0007 | 0.0000 | dense_balanced |
| `block_007.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 1.662042 | 0.052864 | 0.0000 | 0.0000 | 0.0007 | dense_balanced |
| `block_007.slot_11.router` | router | f32 | `(6144, 8)` | 0.005887 | 0.000035 | 0.0000 | 0.1395 | 0.0005 | dense_balanced |
| `block_008.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.744364 | 1072.186512 | 0.0121 | 0.0371 | 0.0000 | dense_balanced |
| `block_008.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.492699 | 1121.760877 | 0.0120 | 0.0363 | 0.0000 | dense_balanced |
| `block_008.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.417812 | 1050.851470 | 0.0129 | 0.0364 | 0.0000 | dense_balanced |
| `block_008.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 29.879340 | 892.774897 | 0.0147 | 0.0428 | 0.0000 | dense_balanced |
| `block_008.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.215581 | 1384.995523 | 0.0109 | 0.0328 | 0.0000 | dense_balanced |
| `block_008.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 30.889621 | 954.162328 | 0.0138 | 0.0411 | 0.0000 | dense_balanced |
| `block_008.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.312259 | 1044.081754 | 0.0137 | 0.0407 | 0.0000 | dense_balanced |
| `block_008.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 1.713994 | 0.080620 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_008.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 1.420799 | 0.058407 | 0.0000 | 0.0000 | 0.0007 | dense_balanced |
| `block_008.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 2.423970 | 0.228633 | 0.0000 | 0.0008 | 0.0000 | dense_balanced |
| `block_008.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 1.736072 | 0.090492 | 0.0000 | 0.0000 | 0.0003 | dense_balanced |
| `block_008.slot_11.router` | router | f32 | `(6144, 8)` | 0.003327 | 0.000011 | 0.0000 | 0.2503 | 0.0005 | dense_balanced |
| `block_009.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.568314 | 1060.660959 | 0.0118 | 0.0368 | 0.0000 | dense_balanced |
| `block_009.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 32.056353 | 1027.608066 | 0.0127 | 0.0384 | 0.0000 | dense_balanced |
| `block_009.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.125140 | 1032.019929 | 0.0126 | 0.0387 | 0.0000 | dense_balanced |
| `block_009.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 29.381846 | 863.276016 | 0.0144 | 0.0437 | 0.0000 | dense_balanced |
| `block_009.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.386270 | 1397.731692 | 0.0110 | 0.0337 | 0.0000 | dense_balanced |
| `block_009.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 29.205638 | 852.969144 | 0.0137 | 0.0416 | 0.0000 | dense_balanced |
| `block_009.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.566009 | 934.280423 | 0.0143 | 0.0428 | 0.0000 | dense_balanced |
| `block_009.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 1.392692 | 0.056692 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_009.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 1.509986 | 0.060940 | 0.0000 | 0.0000 | 0.0002 | dense_balanced |
| `block_009.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.259621 | 0.002028 | 0.0000 | 0.0007 | 0.0000 | dense_balanced |
| `block_009.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 1.870248 | 0.072634 | 0.0000 | 0.0000 | 0.0010 | dense_balanced |
| `block_009.slot_11.router` | router | f32 | `(6144, 8)` | 0.034419 | 0.001185 | 0.0000 | 0.0251 | 0.0007 | dense_balanced |
| `block_010.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.328459 | 1110.761891 | 0.0122 | 0.0365 | 0.0000 | dense_balanced |
| `block_010.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.456002 | 1119.302659 | 0.0128 | 0.0369 | 0.0000 | dense_balanced |
| `block_010.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.058663 | 1027.726565 | 0.0124 | 0.0370 | 0.0000 | dense_balanced |
| `block_010.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 29.583361 | 875.128614 | 0.0145 | 0.0443 | 0.0000 | dense_balanced |
| `block_010.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.159172 | 1380.794137 | 0.0108 | 0.0340 | 0.0000 | dense_balanced |
| `block_010.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 29.042923 | 843.482200 | 0.0147 | 0.0441 | 0.0000 | dense_balanced |
| `block_010.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.670326 | 1002.953274 | 0.0139 | 0.0417 | 0.0000 | dense_balanced |
| `block_010.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 5.118065 | 0.529614 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_010.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 1.620535 | 0.075709 | 0.0000 | 0.0000 | 0.0008 | dense_balanced |
| `block_010.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.246541 | 0.001795 | 0.0000 | 0.0008 | 0.0000 | dense_balanced |
| `block_010.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 2.001553 | 0.133307 | 0.0000 | 0.0000 | 0.0003 | dense_balanced |
| `block_010.slot_11.router` | router | f32 | `(6144, 8)` | 0.032336 | 0.001046 | 0.0000 | 0.0263 | 0.0003 | dense_balanced |
| `block_011.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.104795 | 1095.918762 | 0.0123 | 0.0366 | 0.0000 | dense_balanced |
| `block_011.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 32.797805 | 1075.687089 | 0.0118 | 0.0378 | 0.0000 | dense_balanced |
| `block_011.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.079124 | 1094.218276 | 0.0120 | 0.0357 | 0.0000 | dense_balanced |
| `block_011.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.310365 | 918.683603 | 0.0145 | 0.0404 | 0.0000 | dense_balanced |
| `block_011.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.937752 | 1364.381798 | 0.0112 | 0.0332 | 0.0000 | dense_balanced |
| `block_011.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.798933 | 1011.159116 | 0.0134 | 0.0388 | 0.0000 | dense_balanced |
| `block_011.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.995248 | 960.665174 | 0.0147 | 0.0427 | 0.0000 | dense_balanced |
| `block_011.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.846710 | 0.011735 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_011.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 1.799362 | 0.077402 | 0.0000 | 0.0000 | 0.0011 | dense_balanced |
| `block_011.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.257961 | 0.001416 | 0.0000 | 0.0013 | 0.0018 | dense_balanced |
| `block_011.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 2.294285 | 0.118974 | 0.0000 | 0.0000 | 0.0011 | dense_balanced |
| `block_011.slot_11.router` | router | f32 | `(6144, 8)` | 0.035850 | 0.001285 | 0.0000 | 0.0238 | 0.0005 | dense_balanced |
| `block_012.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.494049 | 1121.850639 | 0.0118 | 0.0359 | 0.0000 | dense_balanced |
| `block_012.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.808649 | 1142.992036 | 0.0128 | 0.0367 | 0.0000 | dense_balanced |
| `block_012.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.091292 | 1095.014578 | 0.0121 | 0.0368 | 0.0000 | dense_balanced |
| `block_012.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 29.867999 | 892.089955 | 0.0149 | 0.0438 | 0.0000 | dense_balanced |
| `block_012.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.480414 | 1404.771217 | 0.0108 | 0.0332 | 0.0000 | dense_balanced |
| `block_012.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 28.773765 | 827.928689 | 0.0147 | 0.0447 | 0.0000 | dense_balanced |
| `block_012.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.767305 | 1009.161343 | 0.0136 | 0.0392 | 0.0000 | dense_balanced |
| `block_012.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.336412 | 0.003752 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_012.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 0.445028 | 0.026887 | 0.0000 | 0.0000 | 0.0013 | dense_balanced |
| `block_012.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.504785 | 0.017655 | 0.0000 | 0.0200 | 0.0000 | dense_balanced |
| `block_012.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 0.547636 | 0.027751 | 0.0000 | 0.0000 | 0.0010 | dense_balanced |
| `block_012.slot_11.router` | router | f32 | `(6144, 8)` | 0.025621 | 0.000656 | 0.0000 | 0.0314 | 0.0004 | dense_balanced |
| `block_013.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.769909 | 1140.401885 | 0.0117 | 0.0348 | 0.0000 | dense_balanced |
| `block_013.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.351256 | 1112.301859 | 0.0117 | 0.0355 | 0.0000 | dense_balanced |
| `block_013.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.525521 | 1057.896269 | 0.0120 | 0.0357 | 0.0000 | dense_balanced |
| `block_013.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.789355 | 947.983064 | 0.0141 | 0.0413 | 0.0000 | dense_balanced |
| `block_013.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.877986 | 1359.980978 | 0.0117 | 0.0343 | 0.0000 | dense_balanced |
| `block_013.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 30.574557 | 934.791059 | 0.0134 | 0.0405 | 0.0000 | dense_balanced |
| `block_013.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.286399 | 1042.409921 | 0.0132 | 0.0407 | 0.0000 | dense_balanced |
| `block_013.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.984840 | 0.011753 | 0.0000 | 0.0000 | 0.0005 | dense_balanced |
| `block_013.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 1.831669 | 0.059645 | 0.0000 | 0.0000 | 0.0020 | dense_balanced |
| `block_013.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.231364 | 0.001251 | 0.0000 | 0.0011 | 0.0020 | dense_balanced |
| `block_013.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 2.339833 | 0.120585 | 0.0000 | 0.0000 | 0.0010 | dense_balanced |
| `block_013.slot_11.router` | router | f32 | `(6144, 8)` | 0.040007 | 0.001600 | 0.0000 | 0.0201 | 0.0005 | dense_balanced |
| `block_014.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.604110 | 1063.026358 | 0.0134 | 0.0378 | 0.0000 | dense_balanced |
| `block_014.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.271735 | 1107.008180 | 0.0122 | 0.0364 | 0.0000 | dense_balanced |
| `block_014.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.650863 | 1132.380553 | 0.0123 | 0.0373 | 0.0000 | dense_balanced |
| `block_014.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.567947 | 996.521707 | 0.0133 | 0.0392 | 0.0000 | dense_balanced |
| `block_014.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.969814 | 1366.765462 | 0.0112 | 0.0336 | 0.0000 | dense_balanced |
| `block_014.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.118444 | 968.349329 | 0.0127 | 0.0384 | 0.0000 | dense_balanced |
| `block_014.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.678324 | 941.135275 | 0.0146 | 0.0430 | 0.0000 | dense_balanced |
| `block_014.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 1.817615 | 0.043807 | 0.0000 | 0.0000 | 0.0010 | dense_balanced |
| `block_014.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.218527 | 0.080758 | 0.0000 | 0.0000 | 0.0024 | dense_balanced |
| `block_014.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.270194 | 0.001392 | 0.0000 | 0.0015 | 0.0028 | dense_balanced |
| `block_014.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 2.476186 | 0.109625 | 0.0000 | 0.0000 | 0.0010 | dense_balanced |
| `block_014.slot_11.router` | router | f32 | `(6144, 8)` | 0.033138 | 0.001098 | 0.0000 | 0.0238 | 0.0002 | dense_balanced |
| `block_015.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.852166 | 1079.263289 | 0.0125 | 0.0373 | 0.0000 | dense_balanced |
| `block_015.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.166791 | 1100.027094 | 0.0124 | 0.0368 | 0.0000 | dense_balanced |
| `block_015.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.414772 | 1050.716693 | 0.0123 | 0.0372 | 0.0000 | dense_balanced |
| `block_015.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.349037 | 982.761621 | 0.0126 | 0.0384 | 0.0000 | dense_balanced |
| `block_015.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.388715 | 1324.073171 | 0.0114 | 0.0348 | 0.0000 | dense_balanced |
| `block_015.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.642509 | 1001.208112 | 0.0122 | 0.0384 | 0.0000 | dense_balanced |
| `block_015.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.130559 | 969.111596 | 0.0148 | 0.0423 | 0.0000 | dense_balanced |
| `block_015.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.990086 | 0.015258 | 0.0000 | 0.0000 | 0.0018 | dense_balanced |
| `block_015.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.129479 | 0.068322 | 0.0000 | 0.0000 | 0.0020 | dense_balanced |
| `block_015.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.403220 | 0.001879 | 0.0000 | 0.0007 | 0.0052 | dense_balanced |
| `block_015.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 2.680646 | 0.120183 | 0.0000 | 0.0000 | 0.0016 | dense_balanced |
| `block_015.slot_11.router` | router | f32 | `(6144, 8)` | 0.020215 | 0.000409 | 0.0000 | 0.0403 | 0.0003 | dense_balanced |
| `block_016.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 31.976729 | 1022.510111 | 0.0132 | 0.0377 | 0.0000 | dense_balanced |
| `block_016.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.863061 | 1146.706842 | 0.0115 | 0.0360 | 0.0000 | dense_balanced |
| `block_016.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.381647 | 1048.568977 | 0.0124 | 0.0375 | 0.0000 | dense_balanced |
| `block_016.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.495385 | 1055.928961 | 0.0137 | 0.0381 | 0.0000 | dense_balanced |
| `block_016.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.612612 | 1340.479209 | 0.0116 | 0.0349 | 0.0000 | dense_balanced |
| `block_016.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 33.019694 | 1090.260705 | 0.0120 | 0.0368 | 0.0000 | dense_balanced |
| `block_016.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.147771 | 970.180954 | 0.0128 | 0.0406 | 0.0000 | dense_balanced |
| `block_016.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 3.546150 | 0.266377 | 0.0000 | 0.0000 | 0.0013 | dense_balanced |
| `block_016.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.678787 | 0.109804 | 0.0000 | 0.0000 | 0.0044 | dense_balanced |
| `block_016.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.258579 | 0.000900 | 0.0000 | 0.0015 | 0.0070 | dense_balanced |
| `block_016.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 2.703091 | 0.084809 | 0.0000 | 0.0000 | 0.0010 | dense_balanced |
| `block_016.slot_11.router` | router | f32 | `(6144, 8)` | 0.032424 | 0.001051 | 0.0000 | 0.0254 | 0.0002 | dense_balanced |
| `block_017.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.277259 | 1041.820080 | 0.0123 | 0.0376 | 0.0000 | dense_balanced |
| `block_017.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.562178 | 1126.404790 | 0.0115 | 0.0346 | 0.0000 | dense_balanced |
| `block_017.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 31.813743 | 1012.111623 | 0.0123 | 0.0382 | 0.0000 | dense_balanced |
| `block_017.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.973647 | 1022.295104 | 0.0132 | 0.0382 | 0.0000 | dense_balanced |
| `block_017.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.029828 | 1371.189693 | 0.0119 | 0.0347 | 0.0000 | dense_balanced |
| `block_017.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.818678 | 1012.411558 | 0.0125 | 0.0387 | 0.0000 | dense_balanced |
| `block_017.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.361061 | 983.515913 | 0.0137 | 0.0409 | 0.0000 | dense_balanced |
| `block_017.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.815083 | 0.023653 | 0.0000 | 0.0000 | 0.0028 | dense_balanced |
| `block_017.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.351334 | 0.056807 | 0.0000 | 0.0000 | 0.0028 | dense_balanced |
| `block_017.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 1.318732 | 0.021572 | 0.0000 | 0.0008 | 0.0067 | dense_balanced |
| `block_017.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 2.824308 | 0.105938 | 0.0000 | 0.0000 | 0.0013 | dense_balanced |
| `block_017.slot_11.router` | router | f32 | `(6144, 8)` | 0.007078 | 0.000050 | 0.0000 | 0.1137 | 0.0002 | dense_balanced |
| `block_018.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.180508 | 1100.932836 | 0.0122 | 0.0369 | 0.0000 | dense_balanced |
| `block_018.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.233480 | 1104.463646 | 0.0118 | 0.0356 | 0.0000 | dense_balanced |
| `block_018.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.671260 | 1133.726797 | 0.0123 | 0.0359 | 0.0000 | dense_balanced |
| `block_018.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.425063 | 987.530012 | 0.0131 | 0.0387 | 0.0000 | dense_balanced |
| `block_018.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.834361 | 1356.765389 | 0.0108 | 0.0330 | 0.0000 | dense_balanced |
| `block_018.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.182158 | 1035.690903 | 0.0138 | 0.0393 | 0.0000 | dense_balanced |
| `block_018.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.378105 | 984.585463 | 0.0131 | 0.0391 | 0.0000 | dense_balanced |
| `block_018.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.525705 | 0.004420 | 0.0000 | 0.0000 | 0.0008 | dense_balanced |
| `block_018.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.357503 | 0.066149 | 0.0000 | 0.0000 | 0.0031 | dense_balanced |
| `block_018.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.349465 | 0.001594 | 0.0000 | 0.0018 | 0.0073 | dense_balanced |
| `block_018.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 2.842006 | 0.090624 | 0.0000 | 0.0000 | 0.0015 | dense_balanced |
| `block_018.slot_11.router` | router | f32 | `(6144, 8)` | 0.027042 | 0.000731 | 0.0000 | 0.0292 | 0.0002 | dense_balanced |
| `block_019.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.742134 | 1072.047101 | 0.0116 | 0.0367 | 0.0000 | dense_balanced |
| `block_019.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.471051 | 1120.239879 | 0.0116 | 0.0364 | 0.0000 | dense_balanced |
| `block_019.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.675701 | 1067.701027 | 0.0119 | 0.0367 | 0.0000 | dense_balanced |
| `block_019.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.011399 | 1024.703455 | 0.0123 | 0.0385 | 0.0000 | dense_balanced |
| `block_019.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.873758 | 1359.667674 | 0.0114 | 0.0334 | 0.0000 | dense_balanced |
| `block_019.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.291258 | 1042.721770 | 0.0121 | 0.0370 | 0.0000 | dense_balanced |
| `block_019.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.890805 | 954.133699 | 0.0134 | 0.0427 | 0.0000 | dense_balanced |
| `block_019.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 1.052098 | 0.024386 | 0.0000 | 0.0000 | 0.0011 | dense_balanced |
| `block_019.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.597108 | 0.119045 | 0.0000 | 0.0000 | 0.0029 | dense_balanced |
| `block_019.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 1.264673 | 0.017818 | 0.0000 | 0.0008 | 0.0072 | dense_balanced |
| `block_019.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 3.120506 | 0.102321 | 0.0000 | 0.0000 | 0.0016 | dense_balanced |
| `block_019.slot_11.router` | router | f32 | `(6144, 8)` | 0.008037 | 0.000065 | 0.0000 | 0.1021 | 0.0005 | dense_balanced |
| `block_020.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.826841 | 1077.583455 | 0.0124 | 0.0384 | 0.0000 | dense_balanced |
| `block_020.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.821649 | 1143.874598 | 0.0121 | 0.0345 | 0.0000 | dense_balanced |
| `block_020.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.001813 | 1089.109431 | 0.0127 | 0.0368 | 0.0000 | dense_balanced |
| `block_020.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 33.045433 | 1091.995969 | 0.0119 | 0.0367 | 0.0000 | dense_balanced |
| `block_020.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.273840 | 1389.306702 | 0.0107 | 0.0336 | 0.0000 | dense_balanced |
| `block_020.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.277835 | 1041.794043 | 0.0127 | 0.0379 | 0.0000 | dense_balanced |
| `block_020.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.389516 | 923.467651 | 0.0140 | 0.0417 | 0.0000 | dense_balanced |
| `block_020.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.732340 | 0.028306 | 0.0000 | 0.0000 | 0.0041 | dense_balanced |
| `block_020.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.724360 | 0.109232 | 0.0000 | 0.0000 | 0.0047 | dense_balanced |
| `block_020.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.272331 | 0.000843 | 0.0000 | 0.0013 | 0.0078 | dense_balanced |
| `block_020.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 2.963708 | 0.088840 | 0.0000 | 0.0000 | 0.0016 | dense_balanced |
| `block_020.slot_11.router` | router | f32 | `(6144, 8)` | 0.038706 | 0.001498 | 0.0000 | 0.0221 | 0.0008 | dense_balanced |
| `block_021.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.996877 | 1155.763908 | 0.0118 | 0.0352 | 0.0000 | dense_balanced |
| `block_021.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.235462 | 1104.595825 | 0.0122 | 0.0368 | 0.0000 | dense_balanced |
| `block_021.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.025840 | 1090.684517 | 0.0120 | 0.0367 | 0.0000 | dense_balanced |
| `block_021.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.061753 | 1027.953847 | 0.0124 | 0.0387 | 0.0000 | dense_balanced |
| `block_021.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.867891 | 1359.240962 | 0.0122 | 0.0349 | 0.0000 | dense_balanced |
| `block_021.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.157809 | 970.796143 | 0.0125 | 0.0394 | 0.0000 | dense_balanced |
| `block_021.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.688845 | 941.776208 | 0.0136 | 0.0403 | 0.0000 | dense_balanced |
| `block_021.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 2.125393 | 0.185111 | 0.0000 | 0.0000 | 0.0059 | dense_balanced |
| `block_021.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 1.893033 | 0.070356 | 0.0000 | 0.0000 | 0.0052 | dense_balanced |
| `block_021.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.460254 | 0.002419 | 0.0000 | 0.0018 | 0.0072 | dense_balanced |
| `block_021.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 2.862234 | 0.077989 | 0.0000 | 0.0000 | 0.0013 | dense_balanced |
| `block_021.slot_11.router` | router | f32 | `(6144, 8)` | 0.022523 | 0.000507 | 0.0000 | 0.0357 | 0.0002 | dense_balanced |
| `block_022.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.795183 | 1075.498787 | 0.0123 | 0.0353 | 0.0000 | dense_balanced |
| `block_022.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.506740 | 1190.697458 | 0.0120 | 0.0348 | 0.0000 | dense_balanced |
| `block_022.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.778592 | 1074.423865 | 0.0123 | 0.0363 | 0.0000 | dense_balanced |
| `block_022.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.106566 | 1030.831321 | 0.0131 | 0.0379 | 0.0000 | dense_balanced |
| `block_022.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.665003 | 1418.610278 | 0.0113 | 0.0333 | 0.0000 | dense_balanced |
| `block_022.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.959081 | 1021.375245 | 0.0127 | 0.0380 | 0.0000 | dense_balanced |
| `block_022.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.623000 | 999.999291 | 0.0123 | 0.0388 | 0.0000 | dense_balanced |
| `block_022.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.627181 | 0.008406 | 0.0000 | 0.0000 | 0.0044 | dense_balanced |
| `block_022.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.039391 | 0.073566 | 0.0000 | 0.0000 | 0.0021 | dense_balanced |
| `block_022.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.608302 | 0.003803 | 0.0000 | 0.0011 | 0.0072 | dense_balanced |
| `block_022.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 3.019965 | 0.084163 | 0.0000 | 0.0000 | 0.0018 | dense_balanced |
| `block_022.slot_11.router` | router | f32 | `(6144, 8)` | 0.018419 | 0.000339 | 0.0000 | 0.0437 | 0.0004 | dense_balanced |
| `block_023.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.689841 | 1068.620153 | 0.0128 | 0.0373 | 0.0000 | dense_balanced |
| `block_023.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.083991 | 1161.718360 | 0.0113 | 0.0352 | 0.0000 | dense_balanced |
| `block_023.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.964812 | 1086.678717 | 0.0119 | 0.0368 | 0.0000 | dense_balanced |
| `block_023.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.472104 | 928.548015 | 0.0141 | 0.0413 | 0.0000 | dense_balanced |
| `block_023.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.552870 | 1410.217859 | 0.0109 | 0.0329 | 0.0000 | dense_balanced |
| `block_023.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 30.625860 | 937.916432 | 0.0137 | 0.0399 | 0.0000 | dense_balanced |
| `block_023.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.266542 | 977.596141 | 0.0144 | 0.0416 | 0.0000 | dense_balanced |
| `block_023.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 2.150004 | 0.139590 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_023.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 0.423627 | 0.019548 | 0.0000 | 0.0002 | 0.0018 | dense_balanced |
| `block_023.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.619056 | 0.024765 | 0.0000 | 0.0094 | 0.0000 | dense_balanced |
| `block_023.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 0.596431 | 0.033110 | 0.0000 | 0.0000 | 0.0007 | dense_balanced |
| `block_023.slot_11.router` | router | f32 | `(6144, 8)` | 0.023406 | 0.000548 | 0.0000 | 0.0381 | 0.0009 | dense_balanced |
| `block_024.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.934730 | 1084.695296 | 0.0133 | 0.0375 | 0.0000 | dense_balanced |
| `block_024.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.520639 | 1123.630846 | 0.0123 | 0.0370 | 0.0000 | dense_balanced |
| `block_024.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.221829 | 1103.681337 | 0.0126 | 0.0368 | 0.0000 | dense_balanced |
| `block_024.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.771317 | 1009.414068 | 0.0124 | 0.0387 | 0.0000 | dense_balanced |
| `block_024.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.403964 | 1399.040444 | 0.0109 | 0.0330 | 0.0000 | dense_balanced |
| `block_024.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 33.162156 | 1099.708832 | 0.0126 | 0.0367 | 0.0000 | dense_balanced |
| `block_024.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.396840 | 923.967836 | 0.0139 | 0.0430 | 0.0000 | dense_balanced |
| `block_024.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 1.596526 | 0.073486 | 0.0000 | 0.0000 | 0.0046 | dense_balanced |
| `block_024.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.671848 | 0.088787 | 0.0000 | 0.0000 | 0.0044 | dense_balanced |
| `block_024.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.915080 | 0.009780 | 0.0000 | 0.0015 | 0.0078 | dense_balanced |
| `block_024.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 2.789244 | 0.073456 | 0.0000 | 0.0000 | 0.0013 | dense_balanced |
| `block_024.slot_11.router` | router | f32 | `(6144, 8)` | 0.012675 | 0.000161 | 0.0000 | 0.0662 | 0.0004 | dense_balanced |
| `block_025.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.138764 | 1098.135143 | 0.0117 | 0.0359 | 0.0000 | dense_balanced |
| `block_025.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.400656 | 1115.603279 | 0.0117 | 0.0367 | 0.0000 | dense_balanced |
| `block_025.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.455902 | 1119.297294 | 0.0120 | 0.0356 | 0.0000 | dense_balanced |
| `block_025.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.090597 | 1029.752344 | 0.0121 | 0.0369 | 0.0000 | dense_balanced |
| `block_025.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.092983 | 1375.880628 | 0.0116 | 0.0339 | 0.0000 | dense_balanced |
| `block_025.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.152745 | 970.492241 | 0.0136 | 0.0394 | 0.0000 | dense_balanced |
| `block_025.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.468781 | 990.276392 | 0.0130 | 0.0395 | 0.0000 | dense_balanced |
| `block_025.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 1.574564 | 0.093191 | 0.0000 | 0.0000 | 0.0057 | dense_balanced |
| `block_025.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 1.699693 | 0.076994 | 0.0000 | 0.0000 | 0.0050 | dense_balanced |
| `block_025.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.354996 | 0.001434 | 0.0000 | 0.0016 | 0.0081 | dense_balanced |
| `block_025.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 3.116599 | 0.099684 | 0.0000 | 0.0000 | 0.0020 | dense_balanced |
| `block_025.slot_11.router` | router | f32 | `(6144, 8)` | 0.032181 | 0.001035 | 0.0000 | 0.0262 | 0.0003 | dense_balanced |
| `block_026.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.009225 | 1089.608872 | 0.0120 | 0.0371 | 0.0000 | dense_balanced |
| `block_026.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.982178 | 1154.777858 | 0.0115 | 0.0352 | 0.0000 | dense_balanced |
| `block_026.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.553464 | 1125.820880 | 0.0126 | 0.0366 | 0.0000 | dense_balanced |
| `block_026.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.222015 | 974.807540 | 0.0129 | 0.0382 | 0.0000 | dense_balanced |
| `block_026.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.589824 | 1412.993531 | 0.0109 | 0.0331 | 0.0000 | dense_balanced |
| `block_026.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.856449 | 1014.831846 | 0.0131 | 0.0379 | 0.0000 | dense_balanced |
| `block_026.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.591905 | 935.858599 | 0.0141 | 0.0419 | 0.0000 | dense_balanced |
| `block_026.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.761766 | 0.014848 | 0.0000 | 0.0000 | 0.0062 | dense_balanced |
| `block_026.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.088587 | 0.088848 | 0.0000 | 0.0000 | 0.0041 | dense_balanced |
| `block_026.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.548738 | 0.003169 | 0.0000 | 0.0018 | 0.0073 | dense_balanced |
| `block_026.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 2.907154 | 0.083046 | 0.0000 | 0.0000 | 0.0015 | dense_balanced |
| `block_026.slot_11.router` | router | f32 | `(6144, 8)` | 0.020008 | 0.000400 | 0.0000 | 0.0397 | 0.0004 | dense_balanced |
| `block_027.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.191597 | 1101.675697 | 0.0115 | 0.0359 | 0.0000 | dense_balanced |
| `block_027.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.006404 | 1156.435150 | 0.0121 | 0.0352 | 0.0000 | dense_balanced |
| `block_027.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.278295 | 1107.444671 | 0.0118 | 0.0370 | 0.0000 | dense_balanced |
| `block_027.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.560824 | 996.053214 | 0.0127 | 0.0384 | 0.0000 | dense_balanced |
| `block_027.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.507881 | 1406.821366 | 0.0117 | 0.0334 | 0.0000 | dense_balanced |
| `block_027.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.936754 | 1019.864571 | 0.0128 | 0.0377 | 0.0000 | dense_balanced |
| `block_027.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.553028 | 995.592784 | 0.0129 | 0.0394 | 0.0000 | dense_balanced |
| `block_027.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 6.217724 | 0.952136 | 0.0000 | 0.0000 | 0.0057 | dense_balanced |
| `block_027.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.021143 | 0.069190 | 0.0000 | 0.0000 | 0.0044 | dense_balanced |
| `block_027.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.464693 | 0.002272 | 0.0000 | 0.0018 | 0.0070 | dense_balanced |
| `block_027.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 2.986758 | 0.085072 | 0.0000 | 0.0000 | 0.0015 | dense_balanced |
| `block_027.slot_11.router` | router | f32 | `(6144, 8)` | 0.024305 | 0.000591 | 0.0000 | 0.0341 | 0.0003 | dense_balanced |
| `block_028.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.709334 | 1069.794195 | 0.0122 | 0.0370 | 0.0000 | dense_balanced |
| `block_028.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.695355 | 1135.372980 | 0.0114 | 0.0342 | 0.0000 | dense_balanced |
| `block_028.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.466278 | 1119.991714 | 0.0116 | 0.0354 | 0.0000 | dense_balanced |
| `block_028.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.347864 | 982.672174 | 0.0135 | 0.0390 | 0.0000 | dense_balanced |
| `block_028.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.352707 | 1395.221390 | 0.0107 | 0.0336 | 0.0000 | dense_balanced |
| `block_028.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.831197 | 1013.218419 | 0.0133 | 0.0383 | 0.0000 | dense_balanced |
| `block_028.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.758718 | 946.092557 | 0.0131 | 0.0415 | 0.0000 | dense_balanced |
| `block_028.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.609200 | 0.010880 | 0.0000 | 0.0000 | 0.0059 | dense_balanced |
| `block_028.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 3.042699 | 0.137997 | 0.0000 | 0.0000 | 0.0034 | dense_balanced |
| `block_028.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 3.545710 | 0.135649 | 0.0000 | 0.0008 | 0.0075 | dense_balanced |
| `block_028.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 3.119395 | 0.108120 | 0.0000 | 0.0000 | 0.0016 | dense_balanced |
| `block_028.slot_11.router` | router | f32 | `(6144, 8)` | 0.003270 | 0.000011 | 0.0000 | 0.2493 | 0.0003 | dense_balanced |
| `block_029.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.610141 | 1063.342560 | 0.0119 | 0.0361 | 0.0000 | dense_balanced |
| `block_029.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.549078 | 1125.517776 | 0.0118 | 0.0363 | 0.0000 | dense_balanced |
| `block_029.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 31.713710 | 1005.757739 | 0.0122 | 0.0387 | 0.0000 | dense_balanced |
| `block_029.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.849216 | 1079.030141 | 0.0120 | 0.0365 | 0.0000 | dense_balanced |
| `block_029.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.384907 | 1397.625129 | 0.0102 | 0.0325 | 0.0000 | dense_balanced |
| `block_029.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.715387 | 1070.268428 | 0.0114 | 0.0379 | 0.0000 | dense_balanced |
| `block_029.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.498431 | 992.137580 | 0.0127 | 0.0387 | 0.0000 | dense_balanced |
| `block_029.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 7.120079 | 2.082098 | 0.0000 | 0.0000 | 0.0036 | dense_balanced |
| `block_029.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.038436 | 0.042627 | 0.0000 | 0.0000 | 0.0033 | dense_balanced |
| `block_029.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.446676 | 0.002032 | 0.0000 | 0.0013 | 0.0067 | dense_balanced |
| `block_029.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 3.084083 | 0.102630 | 0.0000 | 0.0000 | 0.0020 | dense_balanced |
| `block_029.slot_11.router` | router | f32 | `(6144, 8)` | 0.026189 | 0.000686 | 0.0000 | 0.0314 | 0.0006 | dense_balanced |
| `block_030.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.169014 | 1100.178895 | 0.0124 | 0.0364 | 0.0000 | dense_balanced |
| `block_030.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.691678 | 1203.506977 | 0.0117 | 0.0356 | 0.0000 | dense_balanced |
| `block_030.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.861512 | 1079.831768 | 0.0130 | 0.0372 | 0.0000 | dense_balanced |
| `block_030.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 28.436102 | 808.608823 | 0.0148 | 0.0446 | 0.0000 | dense_balanced |
| `block_030.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.015788 | 1297.082131 | 0.0117 | 0.0356 | 0.0000 | dense_balanced |
| `block_030.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 28.410188 | 807.136153 | 0.0148 | 0.0464 | 0.0000 | dense_balanced |
| `block_030.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 29.626657 | 877.734558 | 0.0146 | 0.0436 | 0.0000 | dense_balanced |
| `block_030.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.527965 | 0.013926 | 0.0000 | 0.0000 | 0.0044 | dense_balanced |
| `block_030.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.070368 | 0.090544 | 0.0000 | 0.0000 | 0.0011 | dense_balanced |
| `block_030.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.445285 | 0.001951 | 0.0000 | 0.0013 | 0.0067 | dense_balanced |
| `block_030.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 2.940360 | 0.100486 | 0.0000 | 0.0000 | 0.0016 | dense_balanced |
| `block_030.slot_11.router` | router | f32 | `(6144, 8)` | 0.025977 | 0.000675 | 0.0000 | 0.0323 | 0.0004 | dense_balanced |
| `block_031.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.840502 | 1078.489407 | 0.0123 | 0.0362 | 0.0000 | dense_balanced |
| `block_031.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.701805 | 1135.810939 | 0.0125 | 0.0364 | 0.0000 | dense_balanced |
| `block_031.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.720571 | 1137.053117 | 0.0121 | 0.0367 | 0.0000 | dense_balanced |
| `block_031.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.591082 | 1062.161693 | 0.0120 | 0.0376 | 0.0000 | dense_balanced |
| `block_031.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.020519 | 1370.518353 | 0.0105 | 0.0327 | 0.0000 | dense_balanced |
| `block_031.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.548673 | 1059.395657 | 0.0129 | 0.0385 | 0.0000 | dense_balanced |
| `block_031.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.143208 | 969.898900 | 0.0131 | 0.0408 | 0.0000 | dense_balanced |
| `block_031.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.922324 | 0.054571 | 0.0000 | 0.0000 | 0.0049 | dense_balanced |
| `block_031.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 1.600867 | 0.097658 | 0.0000 | 0.0000 | 0.0039 | dense_balanced |
| `block_031.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.451194 | 0.002105 | 0.0000 | 0.0013 | 0.0067 | dense_balanced |
| `block_031.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 3.296688 | 0.129418 | 0.0000 | 0.0000 | 0.0020 | dense_balanced |
| `block_031.slot_11.router` | router | f32 | `(6144, 8)` | 0.026162 | 0.000684 | 0.0000 | 0.0335 | 0.0006 | dense_balanced |
| `block_032.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.562687 | 1126.452869 | 0.0114 | 0.0353 | 0.0000 | dense_balanced |
| `block_032.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.710673 | 1136.409197 | 0.0118 | 0.0356 | 0.0000 | dense_balanced |
| `block_032.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.645763 | 1132.035079 | 0.0123 | 0.0354 | 0.0000 | dense_balanced |
| `block_032.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.235135 | 975.632119 | 0.0130 | 0.0389 | 0.0000 | dense_balanced |
| `block_032.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.967921 | 1366.556114 | 0.0115 | 0.0340 | 0.0000 | dense_balanced |
| `block_032.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.657819 | 1066.532132 | 0.0119 | 0.0362 | 0.0000 | dense_balanced |
| `block_032.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.421304 | 1051.122900 | 0.0126 | 0.0385 | 0.0000 | dense_balanced |
| `block_032.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.626620 | 0.013691 | 0.0000 | 0.0000 | 0.0081 | dense_balanced |
| `block_032.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.319142 | 0.064320 | 0.0000 | 0.0000 | 0.0046 | dense_balanced |
| `block_032.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.289386 | 0.000875 | 0.0000 | 0.0015 | 0.0073 | dense_balanced |
| `block_032.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 3.481169 | 0.165893 | 0.0000 | 0.0000 | 0.0026 | dense_balanced |
| `block_032.slot_11.router` | router | f32 | `(6144, 8)` | 0.038671 | 0.001495 | 0.0000 | 0.0225 | 0.0003 | dense_balanced |
| `block_033.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.224166 | 1103.799511 | 0.0117 | 0.0354 | 0.0000 | dense_balanced |
| `block_033.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.118109 | 1096.803748 | 0.0121 | 0.0367 | 0.0000 | dense_balanced |
| `block_033.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.663328 | 1066.887302 | 0.0121 | 0.0373 | 0.0000 | dense_balanced |
| `block_033.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.201302 | 1036.922095 | 0.0128 | 0.0384 | 0.0000 | dense_balanced |
| `block_033.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.690789 | 1346.135586 | 0.0117 | 0.0336 | 0.0000 | dense_balanced |
| `block_033.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.044045 | 963.730020 | 0.0140 | 0.0405 | 0.0000 | dense_balanced |
| `block_033.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.142169 | 908.546689 | 0.0137 | 0.0422 | 0.0000 | dense_balanced |
| `block_033.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.406750 | 0.008568 | 0.0000 | 0.0000 | 0.0059 | dense_balanced |
| `block_033.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.356806 | 0.078375 | 0.0000 | 0.0000 | 0.0033 | dense_balanced |
| `block_033.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.405878 | 0.001607 | 0.0000 | 0.0007 | 0.0065 | dense_balanced |
| `block_033.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 3.663123 | 0.179585 | 0.0000 | 0.0000 | 0.0024 | dense_balanced |
| `block_033.slot_11.router` | router | f32 | `(6144, 8)` | 0.028646 | 0.000820 | 0.0000 | 0.0300 | 0.0005 | dense_balanced |
| `block_034.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.871338 | 1080.524418 | 0.0117 | 0.0362 | 0.0000 | dense_balanced |
| `block_034.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.073878 | 1161.027049 | 0.0114 | 0.0346 | 0.0000 | dense_balanced |
| `block_034.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.095726 | 1095.283608 | 0.0119 | 0.0361 | 0.0000 | dense_balanced |
| `block_034.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.671919 | 940.747470 | 0.0133 | 0.0405 | 0.0000 | dense_balanced |
| `block_034.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.533975 | 1408.547543 | 0.0113 | 0.0328 | 0.0000 | dense_balanced |
| `block_034.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.221682 | 974.791314 | 0.0132 | 0.0388 | 0.0000 | dense_balanced |
| `block_034.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.674037 | 1067.587626 | 0.0125 | 0.0381 | 0.0000 | dense_balanced |
| `block_034.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.400159 | 0.004272 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_034.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 0.369949 | 0.020188 | 0.0000 | 0.0003 | 0.0037 | dense_balanced |
| `block_034.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.767153 | 0.038783 | 0.0000 | 0.0041 | 0.0000 | dense_balanced |
| `block_034.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 0.597203 | 0.040077 | 0.0000 | 0.0000 | 0.0010 | dense_balanced |
| `block_034.slot_11.router` | router | f32 | `(6144, 8)` | 0.018766 | 0.000352 | 0.0000 | 0.0466 | 0.0015 | dense_balanced |
| `block_035.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.740074 | 1071.912033 | 0.0122 | 0.0373 | 0.0000 | dense_balanced |
| `block_035.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.887237 | 1148.343917 | 0.0112 | 0.0342 | 0.0000 | dense_balanced |
| `block_035.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.334665 | 1111.140724 | 0.0127 | 0.0367 | 0.0000 | dense_balanced |
| `block_035.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.989090 | 1023.301698 | 0.0139 | 0.0389 | 0.0000 | dense_balanced |
| `block_035.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.123671 | 1378.121057 | 0.0112 | 0.0348 | 0.0000 | dense_balanced |
| `block_035.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.540564 | 1058.876578 | 0.0127 | 0.0387 | 0.0000 | dense_balanced |
| `block_035.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.511483 | 1056.959908 | 0.0135 | 0.0391 | 0.0000 | dense_balanced |
| `block_035.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 1.274332 | 0.112892 | 0.0000 | 0.0000 | 0.0073 | dense_balanced |
| `block_035.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 1.854669 | 0.107690 | 0.0000 | 0.0000 | 0.0050 | dense_balanced |
| `block_035.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.701733 | 0.005159 | 0.0000 | 0.0015 | 0.0065 | dense_balanced |
| `block_035.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 3.507635 | 0.182671 | 0.0000 | 0.0000 | 0.0023 | dense_balanced |
| `block_035.slot_11.router` | router | f32 | `(6144, 8)` | 0.016309 | 0.000266 | 0.0000 | 0.0531 | 0.0004 | dense_balanced |
| `block_036.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.232890 | 1104.424965 | 0.0113 | 0.0354 | 0.0000 | dense_balanced |
| `block_036.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.058352 | 1159.971316 | 0.0124 | 0.0354 | 0.0000 | dense_balanced |
| `block_036.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.889960 | 1081.749481 | 0.0125 | 0.0363 | 0.0000 | dense_balanced |
| `block_036.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.825016 | 1012.831081 | 0.0130 | 0.0381 | 0.0000 | dense_balanced |
| `block_036.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.358896 | 1395.671387 | 0.0108 | 0.0332 | 0.0000 | dense_balanced |
| `block_036.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.599921 | 1062.754791 | 0.0126 | 0.0368 | 0.0000 | dense_balanced |
| `block_036.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.614120 | 937.223868 | 0.0135 | 0.0403 | 0.0000 | dense_balanced |
| `block_036.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.776642 | 0.023329 | 0.0000 | 0.0000 | 0.0063 | dense_balanced |
| `block_036.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.923167 | 0.118768 | 0.0000 | 0.0000 | 0.0047 | dense_balanced |
| `block_036.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.570650 | 0.003437 | 0.0000 | 0.0016 | 0.0063 | dense_balanced |
| `block_036.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 3.524575 | 0.195974 | 0.0000 | 0.0000 | 0.0021 | dense_balanced |
| `block_036.slot_11.router` | router | f32 | `(6144, 8)` | 0.019277 | 0.000371 | 0.0000 | 0.0458 | 0.0003 | dense_balanced |
| `block_037.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.918050 | 1083.595682 | 0.0122 | 0.0370 | 0.0000 | dense_balanced |
| `block_037.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.844266 | 1145.363129 | 0.0117 | 0.0344 | 0.0000 | dense_balanced |
| `block_037.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.984137 | 1087.952708 | 0.0128 | 0.0376 | 0.0000 | dense_balanced |
| `block_037.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.517117 | 993.328596 | 0.0127 | 0.0385 | 0.0000 | dense_balanced |
| `block_037.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.257949 | 1388.101697 | 0.0117 | 0.0341 | 0.0000 | dense_balanced |
| `block_037.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.820096 | 1012.502853 | 0.0130 | 0.0385 | 0.0000 | dense_balanced |
| `block_037.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.663544 | 1002.579803 | 0.0132 | 0.0409 | 0.0000 | dense_balanced |
| `block_037.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.613879 | 0.018616 | 0.0000 | 0.0000 | 0.0049 | dense_balanced |
| `block_037.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.148947 | 0.120943 | 0.0000 | 0.0000 | 0.0015 | dense_balanced |
| `block_037.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.383154 | 0.001583 | 0.0000 | 0.0018 | 0.0060 | dense_balanced |
| `block_037.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 3.401030 | 0.189905 | 0.0000 | 0.0000 | 0.0021 | dense_balanced |
| `block_037.slot_11.router` | router | f32 | `(6144, 8)` | 0.029251 | 0.000855 | 0.0000 | 0.0286 | 0.0003 | dense_balanced |
| `block_038.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.378543 | 1114.053846 | 0.0120 | 0.0361 | 0.0000 | dense_balanced |
| `block_038.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.013576 | 1156.922783 | 0.0118 | 0.0357 | 0.0000 | dense_balanced |
| `block_038.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.776695 | 1140.852298 | 0.0119 | 0.0346 | 0.0000 | dense_balanced |
| `block_038.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.887481 | 954.010238 | 0.0132 | 0.0408 | 0.0000 | dense_balanced |
| `block_038.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.526595 | 1334.182922 | 0.0119 | 0.0348 | 0.0000 | dense_balanced |
| `block_038.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.966649 | 1021.846181 | 0.0128 | 0.0383 | 0.0000 | dense_balanced |
| `block_038.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.921285 | 956.120966 | 0.0140 | 0.0409 | 0.0000 | dense_balanced |
| `block_038.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.764259 | 0.026122 | 0.0000 | 0.0000 | 0.0057 | dense_balanced |
| `block_038.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.569559 | 0.094045 | 0.0000 | 0.0000 | 0.0036 | dense_balanced |
| `block_038.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 2.318477 | 0.056854 | 0.0000 | 0.0010 | 0.0057 | dense_balanced |
| `block_038.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 3.740998 | 0.226490 | 0.0000 | 0.0000 | 0.0028 | dense_balanced |
| `block_038.slot_11.router` | router | f32 | `(6144, 8)` | 0.004856 | 0.000024 | 0.0000 | 0.1741 | 0.0005 | dense_balanced |
| `block_039.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.394549 | 1049.401804 | 0.0129 | 0.0383 | 0.0000 | dense_balanced |
| `block_039.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.077679 | 1161.288130 | 0.0116 | 0.0346 | 0.0000 | dense_balanced |
| `block_039.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 31.923080 | 1019.082751 | 0.0126 | 0.0385 | 0.0000 | dense_balanced |
| `block_039.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.251104 | 1040.133652 | 0.0120 | 0.0376 | 0.0000 | dense_balanced |
| `block_039.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.147394 | 1379.923482 | 0.0113 | 0.0337 | 0.0000 | dense_balanced |
| `block_039.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.191802 | 1036.300926 | 0.0124 | 0.0380 | 0.0000 | dense_balanced |
| `block_039.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.488557 | 929.506067 | 0.0149 | 0.0424 | 0.0000 | dense_balanced |
| `block_039.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 1.686280 | 0.242475 | 0.0000 | 0.0000 | 0.0055 | dense_balanced |
| `block_039.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 3.901183 | 0.295706 | 0.0000 | 0.0000 | 0.0028 | dense_balanced |
| `block_039.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.398533 | 0.001646 | 0.0000 | 0.0015 | 0.0060 | dense_balanced |
| `block_039.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 3.693985 | 0.221936 | 0.0000 | 0.0000 | 0.0036 | dense_balanced |
| `block_039.slot_11.router` | router | f32 | `(6144, 8)` | 0.028070 | 0.000788 | 0.0000 | 0.0304 | 0.0005 | dense_balanced |
| `block_040.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.549828 | 1059.491199 | 0.0125 | 0.0381 | 0.0000 | dense_balanced |
| `block_040.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.186607 | 1168.723584 | 0.0119 | 0.0346 | 0.0000 | dense_balanced |
| `block_040.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.950777 | 1085.752754 | 0.0125 | 0.0378 | 0.0000 | dense_balanced |
| `block_040.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.761022 | 1008.753325 | 0.0134 | 0.0399 | 0.0000 | dense_balanced |
| `block_040.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.275284 | 1389.446497 | 0.0112 | 0.0334 | 0.0000 | dense_balanced |
| `block_040.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 30.883034 | 953.760162 | 0.0125 | 0.0386 | 0.0000 | dense_balanced |
| `block_040.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.649099 | 939.364603 | 0.0148 | 0.0420 | 0.0000 | dense_balanced |
| `block_040.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.301616 | 0.007211 | 0.0000 | 0.0000 | 0.0067 | dense_balanced |
| `block_040.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 1.817321 | 0.119264 | 0.0000 | 0.0000 | 0.0046 | dense_balanced |
| `block_040.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.400808 | 0.001609 | 0.0000 | 0.0013 | 0.0055 | dense_balanced |
| `block_040.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 3.988266 | 0.261965 | 0.0000 | 0.0000 | 0.0033 | dense_balanced |
| `block_040.slot_11.router` | router | f32 | `(6144, 8)` | 0.028219 | 0.000796 | 0.0000 | 0.0311 | 0.0005 | dense_balanced |
| `block_041.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.511835 | 1123.040789 | 0.0123 | 0.0357 | 0.0000 | dense_balanced |
| `block_041.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.141181 | 1165.614214 | 0.0120 | 0.0357 | 0.0000 | dense_balanced |
| `block_041.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.681476 | 1134.433585 | 0.0113 | 0.0349 | 0.0000 | dense_balanced |
| `block_041.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.143039 | 1033.166162 | 0.0127 | 0.0378 | 0.0000 | dense_balanced |
| `block_041.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.249191 | 1387.448893 | 0.0110 | 0.0336 | 0.0000 | dense_balanced |
| `block_041.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.688565 | 1004.117145 | 0.0128 | 0.0391 | 0.0000 | dense_balanced |
| `block_041.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.860332 | 952.311492 | 0.0141 | 0.0407 | 0.0000 | dense_balanced |
| `block_041.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.825750 | 0.047193 | 0.0000 | 0.0000 | 0.0063 | dense_balanced |
| `block_041.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.910306 | 0.120362 | 0.0000 | 0.0000 | 0.0039 | dense_balanced |
| `block_041.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 1.055640 | 0.011953 | 0.0000 | 0.0013 | 0.0055 | dense_balanced |
| `block_041.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 3.963928 | 0.253024 | 0.0000 | 0.0000 | 0.0024 | dense_balanced |
| `block_041.slot_11.router` | router | f32 | `(6144, 8)` | 0.010603 | 0.000112 | 0.0000 | 0.0829 | 0.0004 | dense_balanced |
| `block_042.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.189687 | 1036.170243 | 0.0134 | 0.0385 | 0.0000 | dense_balanced |
| `block_042.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.708219 | 1136.240204 | 0.0117 | 0.0362 | 0.0000 | dense_balanced |
| `block_042.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.225455 | 1103.899668 | 0.0119 | 0.0361 | 0.0000 | dense_balanced |
| `block_042.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.182528 | 1035.714904 | 0.0129 | 0.0377 | 0.0000 | dense_balanced |
| `block_042.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.260405 | 1388.316872 | 0.0111 | 0.0337 | 0.0000 | dense_balanced |
| `block_042.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.576839 | 1061.250383 | 0.0130 | 0.0379 | 0.0000 | dense_balanced |
| `block_042.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.559880 | 933.889196 | 0.0141 | 0.0416 | 0.0000 | dense_balanced |
| `block_042.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.581318 | 0.024264 | 0.0000 | 0.0000 | 0.0063 | dense_balanced |
| `block_042.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 3.910948 | 0.315171 | 0.0000 | 0.0000 | 0.0010 | dense_balanced |
| `block_042.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 1.419465 | 0.022559 | 0.0000 | 0.0011 | 0.0054 | dense_balanced |
| `block_042.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 4.204409 | 0.311208 | 0.0000 | 0.0000 | 0.0034 | dense_balanced |
| `block_042.slot_11.router` | router | f32 | `(6144, 8)` | 0.007485 | 0.000056 | 0.0000 | 0.1140 | 0.0002 | dense_balanced |
| `block_043.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.181939 | 1101.034462 | 0.0125 | 0.0362 | 0.0000 | dense_balanced |
| `block_043.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.041716 | 1158.838205 | 0.0122 | 0.0362 | 0.0000 | dense_balanced |
| `block_043.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.915604 | 1083.436708 | 0.0121 | 0.0361 | 0.0000 | dense_balanced |
| `block_043.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.352935 | 921.300191 | 0.0136 | 0.0409 | 0.0000 | dense_balanced |
| `block_043.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.780147 | 1352.774710 | 0.0120 | 0.0347 | 0.0000 | dense_balanced |
| `block_043.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 30.290484 | 917.385420 | 0.0135 | 0.0404 | 0.0000 | dense_balanced |
| `block_043.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.424474 | 925.640399 | 0.0137 | 0.0403 | 0.0000 | dense_balanced |
| `block_043.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.448308 | 0.015264 | 0.0000 | 0.0000 | 0.0041 | dense_balanced |
| `block_043.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 1.853058 | 0.134796 | 0.0000 | 0.0000 | 0.0031 | dense_balanced |
| `block_043.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.709401 | 0.005794 | 0.0000 | 0.0013 | 0.0052 | dense_balanced |
| `block_043.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 4.381602 | 0.371107 | 0.0000 | 0.0000 | 0.0033 | dense_balanced |
| `block_043.slot_11.router` | router | f32 | `(6144, 8)` | 0.014624 | 0.000214 | 0.0000 | 0.0603 | 0.0001 | dense_balanced |
| `block_044.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.025741 | 1090.696811 | 0.0129 | 0.0382 | 0.0000 | dense_balanced |
| `block_044.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.686939 | 1134.785747 | 0.0114 | 0.0348 | 0.0000 | dense_balanced |
| `block_044.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.810139 | 1076.434400 | 0.0125 | 0.0373 | 0.0000 | dense_balanced |
| `block_044.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.179887 | 1035.542647 | 0.0127 | 0.0381 | 0.0000 | dense_balanced |
| `block_044.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.157718 | 1380.681641 | 0.0112 | 0.0344 | 0.0000 | dense_balanced |
| `block_044.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.587697 | 997.781988 | 0.0137 | 0.0397 | 0.0000 | dense_balanced |
| `block_044.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 29.737431 | 884.314160 | 0.0147 | 0.0441 | 0.0000 | dense_balanced |
| `block_044.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 1.287839 | 0.089762 | 0.0000 | 0.0000 | 0.0063 | dense_balanced |
| `block_044.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 3.046447 | 0.147533 | 0.0000 | 0.0000 | 0.0031 | dense_balanced |
| `block_044.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.614603 | 0.004229 | 0.0000 | 0.0013 | 0.0047 | dense_balanced |
| `block_044.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 4.599718 | 0.388508 | 0.0000 | 0.0000 | 0.0039 | dense_balanced |
| `block_044.slot_11.router` | router | f32 | `(6144, 8)` | 0.016473 | 0.000271 | 0.0000 | 0.0528 | 0.0002 | dense_balanced |
| `block_045.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.308830 | 1109.475232 | 0.0122 | 0.0365 | 0.0000 | dense_balanced |
| `block_045.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.304912 | 1109.196533 | 0.0117 | 0.0355 | 0.0000 | dense_balanced |
| `block_045.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 31.942831 | 1020.325193 | 0.0124 | 0.0373 | 0.0000 | dense_balanced |
| `block_045.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.967744 | 1021.923176 | 0.0134 | 0.0396 | 0.0000 | dense_balanced |
| `block_045.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.542414 | 1409.430221 | 0.0113 | 0.0331 | 0.0000 | dense_balanced |
| `block_045.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.565874 | 996.369425 | 0.0130 | 0.0390 | 0.0000 | dense_balanced |
| `block_045.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.355667 | 1046.879797 | 0.0124 | 0.0373 | 0.0000 | dense_balanced |
| `block_045.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 2.054696 | 0.149420 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_045.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 0.490506 | 0.040891 | 0.0000 | 0.0000 | 0.0034 | dense_balanced |
| `block_045.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.506126 | 0.018626 | 0.0000 | 0.0042 | 0.0000 | dense_balanced |
| `block_045.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 0.695722 | 0.046642 | 0.0000 | 0.0000 | 0.0003 | dense_balanced |
| `block_045.slot_11.router` | router | f32 | `(6144, 8)` | 0.027272 | 0.000744 | 0.0000 | 0.0338 | 0.0009 | dense_balanced |
| `block_046.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.360936 | 1047.210670 | 0.0129 | 0.0373 | 0.0000 | dense_balanced |
| `block_046.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.599474 | 1128.916794 | 0.0125 | 0.0371 | 0.0000 | dense_balanced |
| `block_046.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.564091 | 1126.533435 | 0.0119 | 0.0364 | 0.0000 | dense_balanced |
| `block_046.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.176118 | 971.930652 | 0.0134 | 0.0397 | 0.0000 | dense_balanced |
| `block_046.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.466894 | 1403.706608 | 0.0110 | 0.0332 | 0.0000 | dense_balanced |
| `block_046.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.913584 | 1018.458747 | 0.0130 | 0.0390 | 0.0000 | dense_balanced |
| `block_046.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.644418 | 939.076553 | 0.0139 | 0.0415 | 0.0000 | dense_balanced |
| `block_046.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.542803 | 0.015397 | 0.0000 | 0.0000 | 0.0057 | dense_balanced |
| `block_046.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 3.307359 | 0.257192 | 0.0000 | 0.0000 | 0.0005 | dense_balanced |
| `block_046.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.417346 | 0.001889 | 0.0000 | 0.0013 | 0.0049 | dense_balanced |
| `block_046.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 4.967838 | 0.434055 | 0.0000 | 0.0000 | 0.0049 | dense_balanced |
| `block_046.slot_11.router` | router | f32 | `(6144, 8)` | 0.024376 | 0.000594 | 0.0000 | 0.0357 | 0.0002 | dense_balanced |
| `block_047.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.592357 | 1062.261746 | 0.0119 | 0.0374 | 0.0000 | dense_balanced |
| `block_047.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.259334 | 1173.697821 | 0.0116 | 0.0363 | 0.0000 | dense_balanced |
| `block_047.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.142692 | 1033.152429 | 0.0125 | 0.0372 | 0.0000 | dense_balanced |
| `block_047.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.492087 | 1055.698380 | 0.0121 | 0.0372 | 0.0000 | dense_balanced |
| `block_047.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.157453 | 1380.655053 | 0.0111 | 0.0338 | 0.0000 | dense_balanced |
| `block_047.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.003881 | 1024.244382 | 0.0129 | 0.0374 | 0.0000 | dense_balanced |
| `block_047.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.706341 | 942.870052 | 0.0138 | 0.0415 | 0.0000 | dense_balanced |
| `block_047.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 1.017772 | 0.038393 | 0.0000 | 0.0000 | 0.0078 | dense_balanced |
| `block_047.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 3.567557 | 0.374629 | 0.0000 | 0.0000 | 0.0002 | dense_balanced |
| `block_047.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.519002 | 0.002636 | 0.0000 | 0.0008 | 0.0037 | dense_balanced |
| `block_047.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 4.624421 | 0.322975 | 0.0000 | 0.0000 | 0.0057 | dense_balanced |
| `block_047.slot_11.router` | router | f32 | `(6144, 8)` | 0.019337 | 0.000374 | 0.0000 | 0.0462 | 0.0003 | dense_balanced |
| `block_048.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.997863 | 1088.850985 | 0.0126 | 0.0364 | 0.0000 | dense_balanced |
| `block_048.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.591780 | 1128.407390 | 0.0123 | 0.0356 | 0.0000 | dense_balanced |
| `block_048.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.459730 | 1053.631110 | 0.0126 | 0.0388 | 0.0000 | dense_balanced |
| `block_048.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.755255 | 945.876199 | 0.0133 | 0.0399 | 0.0000 | dense_balanced |
| `block_048.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.100945 | 1376.463965 | 0.0117 | 0.0343 | 0.0000 | dense_balanced |
| `block_048.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.085482 | 966.275796 | 0.0133 | 0.0389 | 0.0000 | dense_balanced |
| `block_048.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.353422 | 983.029910 | 0.0136 | 0.0404 | 0.0000 | dense_balanced |
| `block_048.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.378866 | 0.008649 | 0.0000 | 0.0000 | 0.0055 | dense_balanced |
| `block_048.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.157439 | 0.118308 | 0.0000 | 0.0000 | 0.0026 | dense_balanced |
| `block_048.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.240833 | 0.000598 | 0.0000 | 0.0011 | 0.0039 | dense_balanced |
| `block_048.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 5.005953 | 0.411966 | 0.0000 | 0.0002 | 0.0055 | dense_balanced |
| `block_048.slot_11.router` | router | f32 | `(6144, 8)` | 0.040380 | 0.001630 | 0.0000 | 0.0208 | 0.0001 | dense_balanced |
| `block_049.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.963146 | 1086.543076 | 0.0118 | 0.0363 | 0.0000 | dense_balanced |
| `block_049.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.600845 | 1129.012166 | 0.0120 | 0.0362 | 0.0000 | dense_balanced |
| `block_049.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.678143 | 1067.859127 | 0.0126 | 0.0369 | 0.0000 | dense_balanced |
| `block_049.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.522564 | 993.671730 | 0.0135 | 0.0397 | 0.0000 | dense_balanced |
| `block_049.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.540125 | 1409.258567 | 0.0109 | 0.0338 | 0.0000 | dense_balanced |
| `block_049.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.020871 | 1025.336145 | 0.0132 | 0.0378 | 0.0000 | dense_balanced |
| `block_049.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.518083 | 993.371793 | 0.0129 | 0.0398 | 0.0000 | dense_balanced |
| `block_049.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.560115 | 0.020128 | 0.0000 | 0.0000 | 0.0065 | dense_balanced |
| `block_049.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.223204 | 0.131493 | 0.0000 | 0.0000 | 0.0020 | dense_balanced |
| `block_049.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 1.435597 | 0.021067 | 0.0000 | 0.0007 | 0.0033 | dense_balanced |
| `block_049.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 5.046049 | 0.374663 | 0.0000 | 0.0000 | 0.0047 | dense_balanced |
| `block_049.slot_11.router` | router | f32 | `(6144, 8)` | 0.006457 | 0.000042 | 0.0000 | 0.1313 | 0.0000 | dense_balanced |
| `block_050.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.245753 | 1105.243171 | 0.0111 | 0.0347 | 0.0000 | dense_balanced |
| `block_050.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.711680 | 1136.440988 | 0.0121 | 0.0359 | 0.0000 | dense_balanced |
| `block_050.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.232830 | 1104.397960 | 0.0118 | 0.0358 | 0.0000 | dense_balanced |
| `block_050.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.372725 | 922.497158 | 0.0128 | 0.0399 | 0.0000 | dense_balanced |
| `block_050.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.290460 | 1390.508248 | 0.0116 | 0.0346 | 0.0000 | dense_balanced |
| `block_050.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 30.666742 | 940.439981 | 0.0133 | 0.0408 | 0.0000 | dense_balanced |
| `block_050.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 29.799286 | 887.995946 | 0.0139 | 0.0406 | 0.0000 | dense_balanced |
| `block_050.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.365896 | 0.009380 | 0.0000 | 0.0000 | 0.0039 | dense_balanced |
| `block_050.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.107629 | 0.107049 | 0.0000 | 0.0000 | 0.0005 | dense_balanced |
| `block_050.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.843787 | 0.007585 | 0.0000 | 0.0005 | 0.0024 | dense_balanced |
| `block_050.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 4.890558 | 0.348344 | 0.0000 | 0.0000 | 0.0052 | dense_balanced |
| `block_050.slot_11.router` | router | f32 | `(6144, 8)` | 0.010545 | 0.000111 | 0.0000 | 0.0802 | 0.0001 | dense_balanced |
| `block_051.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.716642 | 1070.366258 | 0.0127 | 0.0378 | 0.0000 | dense_balanced |
| `block_051.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.680739 | 1134.308825 | 0.0127 | 0.0373 | 0.0000 | dense_balanced |
| `block_051.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.732493 | 1071.415933 | 0.0124 | 0.0372 | 0.0000 | dense_balanced |
| `block_051.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 26.337809 | 693.674235 | 0.0193 | 0.0570 | 0.0000 | dense_balanced |
| `block_051.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.817442 | 1355.523921 | 0.0112 | 0.0342 | 0.0000 | dense_balanced |
| `block_051.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 30.101025 | 906.063360 | 0.0137 | 0.0422 | 0.0000 | dense_balanced |
| `block_051.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 29.995338 | 899.689000 | 0.0139 | 0.0419 | 0.0000 | dense_balanced |
| `block_051.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.461338 | 0.014625 | 0.0000 | 0.0000 | 0.0059 | dense_balanced |
| `block_051.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.156535 | 0.181492 | 0.0000 | 0.0000 | 0.0002 | dense_balanced |
| `block_051.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.996384 | 0.008903 | 0.0000 | 0.0003 | 0.0026 | dense_balanced |
| `block_051.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 5.250881 | 0.403887 | 0.0000 | 0.0000 | 0.0065 | dense_balanced |
| `block_051.slot_11.router` | router | f32 | `(6144, 8)` | 0.008724 | 0.000076 | 0.0000 | 0.0943 | 0.0001 | dense_balanced |
| `block_052.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.207825 | 1102.656386 | 0.0114 | 0.0357 | 0.0000 | dense_balanced |
| `block_052.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.521397 | 1123.679837 | 0.0114 | 0.0357 | 0.0000 | dense_balanced |
| `block_052.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.361575 | 1112.992994 | 0.0122 | 0.0361 | 0.0000 | dense_balanced |
| `block_052.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.358754 | 983.345378 | 0.0131 | 0.0381 | 0.0000 | dense_balanced |
| `block_052.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.664783 | 1344.291457 | 0.0113 | 0.0335 | 0.0000 | dense_balanced |
| `block_052.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.969090 | 1022.022553 | 0.0127 | 0.0374 | 0.0000 | dense_balanced |
| `block_052.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.091998 | 966.712249 | 0.0128 | 0.0386 | 0.0000 | dense_balanced |
| `block_052.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.356007 | 0.009907 | 0.0000 | 0.0000 | 0.0046 | dense_balanced |
| `block_052.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.271777 | 0.147072 | 0.0000 | 0.0000 | 0.0008 | dense_balanced |
| `block_052.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.318940 | 0.000771 | 0.0000 | 0.0005 | 0.0028 | dense_balanced |
| `block_052.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 5.953707 | 0.561017 | 0.0000 | 0.0000 | 0.0086 | dense_balanced |
| `block_052.slot_11.router` | router | f32 | `(6144, 8)` | 0.027549 | 0.000759 | 0.0000 | 0.0318 | 0.0001 | dense_balanced |
| `block_053.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.484273 | 1055.227051 | 0.0122 | 0.0365 | 0.0000 | dense_balanced |
| `block_053.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.679958 | 1134.333315 | 0.0120 | 0.0369 | 0.0000 | dense_balanced |
| `block_053.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.183959 | 1101.140918 | 0.0125 | 0.0375 | 0.0000 | dense_balanced |
| `block_053.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 28.130795 | 791.333328 | 0.0147 | 0.0469 | 0.0000 | dense_balanced |
| `block_053.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.508944 | 1332.892194 | 0.0119 | 0.0350 | 0.0000 | dense_balanced |
| `block_053.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.044988 | 1026.876507 | 0.0128 | 0.0395 | 0.0000 | dense_balanced |
| `block_053.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.077784 | 965.828441 | 0.0143 | 0.0414 | 0.0000 | dense_balanced |
| `block_053.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.565806 | 0.021819 | 0.0000 | 0.0000 | 0.0046 | dense_balanced |
| `block_053.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.603889 | 0.167880 | 0.0000 | 0.0000 | 0.0008 | dense_balanced |
| `block_053.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.707330 | 0.003555 | 0.0000 | 0.0003 | 0.0026 | dense_balanced |
| `block_053.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 6.068496 | 0.545041 | 0.0000 | 0.0000 | 0.0094 | dense_balanced |
| `block_053.slot_11.router` | router | f32 | `(6144, 8)` | 0.012246 | 0.000150 | 0.0000 | 0.0666 | 0.0001 | dense_balanced |
| `block_054.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.492759 | 1121.753022 | 0.0121 | 0.0352 | 0.0000 | dense_balanced |
| `block_054.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.440011 | 1186.101556 | 0.0112 | 0.0353 | 0.0000 | dense_balanced |
| `block_054.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.366739 | 1113.338101 | 0.0126 | 0.0366 | 0.0000 | dense_balanced |
| `block_054.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 28.730196 | 825.423477 | 0.0142 | 0.0431 | 0.0000 | dense_balanced |
| `block_054.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.557397 | 1336.438849 | 0.0115 | 0.0356 | 0.0000 | dense_balanced |
| `block_054.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.313270 | 980.507055 | 0.0128 | 0.0392 | 0.0000 | dense_balanced |
| `block_054.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.300810 | 979.714463 | 0.0148 | 0.0420 | 0.0000 | dense_balanced |
| `block_054.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.328959 | 0.006350 | 0.0000 | 0.0000 | 0.0067 | dense_balanced |
| `block_054.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 4.008798 | 0.446102 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_054.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.387679 | 0.001148 | 0.0000 | 0.0005 | 0.0024 | dense_balanced |
| `block_054.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 6.447156 | 0.561436 | 0.0000 | 0.0000 | 0.0086 | dense_balanced |
| `block_054.slot_11.router` | router | f32 | `(6144, 8)` | 0.020399 | 0.000416 | 0.0000 | 0.0407 | 0.0001 | dense_balanced |
| `block_055.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.992674 | 1088.514719 | 0.0125 | 0.0369 | 0.0000 | dense_balanced |
| `block_055.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.911859 | 1150.009599 | 0.0118 | 0.0359 | 0.0000 | dense_balanced |
| `block_055.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.130328 | 1097.617422 | 0.0115 | 0.0361 | 0.0000 | dense_balanced |
| `block_055.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.276502 | 1041.728960 | 0.0129 | 0.0378 | 0.0000 | dense_balanced |
| `block_055.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.703693 | 1347.160774 | 0.0114 | 0.0352 | 0.0000 | dense_balanced |
| `block_055.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.267500 | 1041.184095 | 0.0122 | 0.0354 | 0.0000 | dense_balanced |
| `block_055.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.746650 | 945.349724 | 0.0139 | 0.0409 | 0.0000 | dense_balanced |
| `block_055.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.351959 | 0.009737 | 0.0000 | 0.0000 | 0.0057 | dense_balanced |
| `block_055.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 2.821664 | 0.165230 | 0.0000 | 0.0000 | 0.0021 | dense_balanced |
| `block_055.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.728314 | 0.003581 | 0.0000 | 0.0005 | 0.0015 | dense_balanced |
| `block_055.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 6.912750 | 0.442471 | 0.0000 | 0.0000 | 0.0080 | dense_balanced |
| `block_055.slot_11.router` | router | f32 | `(6144, 8)` | 0.011908 | 0.000142 | 0.0000 | 0.0671 | 0.0003 | dense_balanced |
| `block_056.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.115482 | 1031.392682 | 0.0130 | 0.0386 | 0.0000 | dense_balanced |
| `block_056.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.039612 | 1158.666699 | 0.0125 | 0.0371 | 0.0000 | dense_balanced |
| `block_056.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 31.786976 | 1010.407961 | 0.0137 | 0.0384 | 0.0000 | dense_balanced |
| `block_056.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.757813 | 1073.074313 | 0.0127 | 0.0385 | 0.0000 | dense_balanced |
| `block_056.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.445551 | 1328.265127 | 0.0116 | 0.0338 | 0.0000 | dense_balanced |
| `block_056.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.634395 | 1064.923284 | 0.0126 | 0.0376 | 0.0000 | dense_balanced |
| `block_056.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.838995 | 1013.696067 | 0.0132 | 0.0386 | 0.0000 | dense_balanced |
| `block_056.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.506113 | 0.007757 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_056.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 0.555817 | 0.048149 | 0.0000 | 0.0000 | 0.0044 | dense_balanced |
| `block_056.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.681782 | 0.038025 | 0.0000 | 0.0024 | 0.0000 | dense_balanced |
| `block_056.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 0.697407 | 0.034158 | 0.0000 | 0.0000 | 0.0007 | dense_balanced |
| `block_056.slot_11.router` | router | f32 | `(6144, 8)` | 0.019299 | 0.000372 | 0.0000 | 0.0443 | 0.0008 | dense_balanced |
| `block_057.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.285647 | 1042.283641 | 0.0128 | 0.0371 | 0.0000 | dense_balanced |
| `block_057.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.593461 | 1128.520606 | 0.0112 | 0.0354 | 0.0000 | dense_balanced |
| `block_057.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.463969 | 1053.887230 | 0.0132 | 0.0380 | 0.0000 | dense_balanced |
| `block_057.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.756628 | 1008.477133 | 0.0136 | 0.0389 | 0.0000 | dense_balanced |
| `block_057.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.504695 | 1332.586486 | 0.0114 | 0.0342 | 0.0000 | dense_balanced |
| `block_057.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.006276 | 1024.376819 | 0.0125 | 0.0392 | 0.0000 | dense_balanced |
| `block_057.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.672315 | 940.762711 | 0.0138 | 0.0416 | 0.0000 | dense_balanced |
| `block_057.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.595861 | 0.023758 | 0.0000 | 0.0000 | 0.0068 | dense_balanced |
| `block_057.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 3.517995 | 0.210490 | 0.0000 | 0.0000 | 0.0007 | dense_balanced |
| `block_057.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 1.009932 | 0.008324 | 0.0000 | 0.0002 | 0.0023 | dense_balanced |
| `block_057.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 7.441967 | 0.570045 | 0.0000 | 0.0000 | 0.0072 | dense_balanced |
| `block_057.slot_11.router` | router | f32 | `(6144, 8)` | 0.008636 | 0.000075 | 0.0000 | 0.0904 | 0.0001 | dense_balanced |
| `block_058.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.797920 | 1075.696426 | 0.0117 | 0.0359 | 0.0000 | dense_balanced |
| `block_058.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.033865 | 1158.245271 | 0.0127 | 0.0358 | 0.0000 | dense_balanced |
| `block_058.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.246213 | 1039.817218 | 0.0122 | 0.0378 | 0.0000 | dense_balanced |
| `block_058.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.754014 | 945.798715 | 0.0131 | 0.0400 | 0.0000 | dense_balanced |
| `block_058.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 35.548785 | 1263.715202 | 0.0118 | 0.0359 | 0.0000 | dense_balanced |
| `block_058.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.019074 | 962.175260 | 0.0135 | 0.0397 | 0.0000 | dense_balanced |
| `block_058.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.020546 | 901.217199 | 0.0138 | 0.0437 | 0.0000 | dense_balanced |
| `block_058.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.493560 | 0.016110 | 0.0000 | 0.0000 | 0.0065 | dense_balanced |
| `block_058.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 5.049409 | 0.365424 | 0.0000 | 0.0000 | 0.0010 | dense_balanced |
| `block_058.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 1.372380 | 0.019130 | 0.0000 | 0.0000 | 0.0028 | dense_balanced |
| `block_058.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 7.536412 | 0.633300 | 0.0000 | 0.0000 | 0.0055 | dense_balanced |
| `block_058.slot_11.router` | router | f32 | `(6144, 8)` | 0.006776 | 0.000046 | 0.0000 | 0.1163 | 0.0000 | dense_balanced |
| `block_059.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.845141 | 1078.795341 | 0.0122 | 0.0359 | 0.0000 | dense_balanced |
| `block_059.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.012193 | 1089.804330 | 0.0127 | 0.0379 | 0.0000 | dense_balanced |
| `block_059.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.483118 | 1121.113257 | 0.0119 | 0.0373 | 0.0000 | dense_balanced |
| `block_059.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.587784 | 997.788099 | 0.0131 | 0.0384 | 0.0000 | dense_balanced |
| `block_059.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.232469 | 1312.723049 | 0.0122 | 0.0352 | 0.0000 | dense_balanced |
| `block_059.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 30.940241 | 957.275722 | 0.0133 | 0.0406 | 0.0000 | dense_balanced |
| `block_059.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.945154 | 957.596905 | 0.0137 | 0.0414 | 0.0000 | dense_balanced |
| `block_059.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 2.695164 | 0.580174 | 0.0000 | 0.0000 | 0.0050 | dense_balanced |
| `block_059.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 4.417190 | 0.262735 | 0.0000 | 0.0000 | 0.0011 | dense_balanced |
| `block_059.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.361932 | 0.001808 | 0.0000 | 0.0000 | 0.0050 | dense_balanced |
| `block_059.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 7.733720 | 0.866094 | 0.0000 | 0.0000 | 0.0023 | dense_balanced |
| `block_059.slot_11.router` | router | f32 | `(6144, 8)` | 0.027676 | 0.000766 | 0.0000 | 0.0286 | 0.0000 | dense_balanced |
| `block_060.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.387261 | 1114.687631 | 0.0119 | 0.0353 | 0.0000 | dense_balanced |
| `block_060.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 32.304226 | 1043.554004 | 0.0142 | 0.0415 | 0.0000 | dense_balanced |
| `block_060.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.528836 | 1058.116547 | 0.0117 | 0.0364 | 0.0000 | dense_balanced |
| `block_060.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.754982 | 945.864841 | 0.0134 | 0.0395 | 0.0000 | dense_balanced |
| `block_060.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 36.094814 | 1302.806019 | 0.0104 | 0.0341 | 0.0000 | dense_balanced |
| `block_060.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.655373 | 1002.008013 | 0.0140 | 0.0401 | 0.0000 | dense_balanced |
| `block_060.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 30.280138 | 916.886586 | 0.0140 | 0.0431 | 0.0000 | dense_balanced |
| `block_060.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.822909 | 0.052635 | 0.0000 | 0.0000 | 0.0042 | dense_balanced |
| `block_060.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 4.200552 | 0.509499 | 0.0000 | 0.0000 | 0.0003 | dense_balanced |
| `block_060.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 1.687150 | 0.093041 | 0.0000 | 0.0000 | 0.0050 | dense_balanced |
| `block_060.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 4.820732 | 0.634994 | 0.0000 | 0.0000 | 0.0018 | dense_balanced |
| `block_060.slot_11.router` | router | f32 | `(6144, 8)` | 0.007846 | 0.000062 | 0.0000 | 0.0980 | 0.0000 | dense_balanced |
| `block_061.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 33.187642 | 1101.407772 | 0.0120 | 0.0356 | 0.0000 | dense_balanced |
| `block_061.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.936691 | 1151.683570 | 0.0123 | 0.0367 | 0.0000 | dense_balanced |
| `block_061.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 33.596371 | 1128.665069 | 0.0128 | 0.0367 | 0.0000 | dense_balanced |
| `block_061.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.337014 | 1045.681694 | 0.0122 | 0.0373 | 0.0000 | dense_balanced |
| `block_061.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.340355 | 1394.184133 | 0.0109 | 0.0337 | 0.0000 | dense_balanced |
| `block_061.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.013058 | 1024.829644 | 0.0125 | 0.0378 | 0.0000 | dense_balanced |
| `block_061.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.602190 | 1062.891742 | 0.0124 | 0.0386 | 0.0000 | dense_balanced |
| `block_061.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.805027 | 0.024851 | 0.0000 | 0.0000 | 0.0000 | dense_balanced |
| `block_061.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 0.600937 | 0.049733 | 0.0000 | 0.0000 | 0.0033 | dense_balanced |
| `block_061.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.362362 | 0.011204 | 0.0000 | 0.0098 | 0.0000 | dense_balanced |
| `block_061.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 0.862954 | 0.051700 | 0.0000 | 0.0000 | 0.0008 | dense_balanced |
| `block_061.slot_11.router` | router | f32 | `(6144, 8)` | 0.036683 | 0.001346 | 0.0000 | 0.0243 | 0.0013 | dense_balanced |
| `block_062.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.600761 | 1062.800949 | 0.0125 | 0.0367 | 0.0000 | dense_balanced |
| `block_062.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 33.406004 | 1115.960226 | 0.0114 | 0.0354 | 0.0000 | dense_balanced |
| `block_062.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 31.874958 | 1016.009231 | 0.0124 | 0.0382 | 0.0000 | dense_balanced |
| `block_062.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.757777 | 1073.053143 | 0.0127 | 0.0377 | 0.0000 | dense_balanced |
| `block_062.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.241895 | 1386.954981 | 0.0114 | 0.0343 | 0.0000 | dense_balanced |
| `block_062.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 32.385679 | 1048.817704 | 0.0130 | 0.0369 | 0.0000 | dense_balanced |
| `block_062.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.550052 | 995.404839 | 0.0141 | 0.0427 | 0.0000 | dense_balanced |
| `block_062.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.610588 | 0.018270 | 0.0000 | 0.0005 | 0.0000 | dense_balanced |
| `block_062.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 0.630911 | 0.038460 | 0.0000 | 0.0000 | 0.0016 | dense_balanced |
| `block_062.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.305501 | 0.007133 | 0.0000 | 0.0033 | 0.0000 | dense_balanced |
| `block_062.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 0.868938 | 0.039470 | 0.0000 | 0.0000 | 0.0007 | dense_balanced |
| `block_062.slot_11.router` | router | f32 | `(6144, 8)` | 0.039434 | 0.001555 | 0.0000 | 0.0208 | 0.0011 | dense_balanced |
| `block_063.slot_00.moe_expert.gate` | moe_expert.gate | int8 | `(8, 6144, 32768)` | 32.176675 | 1035.323579 | 0.0124 | 0.0383 | 0.0000 | dense_balanced |
| `block_063.slot_01.moe_expert.down` | moe_expert.down | int8 | `(8, 32768, 6144)` | 34.028148 | 1157.905910 | 0.0124 | 0.0365 | 0.0000 | dense_balanced |
| `block_063.slot_02.moe_expert.up` | moe_expert.up | int8 | `(8, 6144, 32768)` | 32.668563 | 1067.230950 | 0.0126 | 0.0368 | 0.0000 | dense_balanced |
| `block_063.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 31.868098 | 1015.573852 | 0.0129 | 0.0387 | 0.0000 | dense_balanced |
| `block_063.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 37.729480 | 1423.501918 | 0.0109 | 0.0328 | 0.0000 | dense_balanced |
| `block_063.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | int8 | `(6144, 6144)` | 31.835068 | 1013.462647 | 0.0126 | 0.0380 | 0.0000 | dense_balanced |
| `block_063.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | int8 | `(6144, 1024)` | 32.406644 | 1050.175151 | 0.0134 | 0.0398 | 0.0000 | dense_balanced |
| `block_063.slot_07.block_norm` | block_norm | f32 | `(6144,)` | 0.937360 | 0.035861 | 0.0000 | 0.0002 | 0.0000 | dense_balanced |
| `block_063.slot_08.block_norm` | block_norm | f32 | `(6144,)` | 0.681916 | 0.039609 | 0.0000 | 0.0000 | 0.0024 | dense_balanced |
| `block_063.slot_09.block_norm` | block_norm | f32 | `(6144,)` | 0.697279 | 0.034568 | 0.0000 | 0.0015 | 0.0000 | dense_balanced |
| `block_063.slot_10.block_norm` | block_norm | f32 | `(6144,)` | 0.990820 | 0.045942 | 0.0000 | 0.0000 | 0.0007 | dense_balanced |
| `block_063.slot_11.router` | router | f32 | `(6144, 8)` | 0.017227 | 0.000297 | 0.0000 | 0.0470 | 0.0009 | dense_balanced |
