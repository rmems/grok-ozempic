# xai-dissect SAAQ-readiness report

- **model_family**: `grok-1`
- **checkpoint**: `/home/raulmc/.models/xai-grok-1/ckpt-0`
- **shards**: 770
- **quantization_candidates**: 448
- **precision_sensitive_tensors**: 257
- **deferred_tensors**: 1
- **routing_critical_tensors**: 64
- **schema_version**: 2

## Quantization candidates

| Rank | Tensor | Kind | Region | Readiness | Opportunity | Risk | Disposition |
| ---: | ------ | ---- | ------ | --------: | ----------: | ---: | ----------- |
| 1 | `block_030.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 2 | `block_056.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 3 | `block_022.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 4 | `block_063.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 5 | `block_058.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 6 | `block_012.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 7 | `block_061.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 8 | `block_047.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 9 | `block_051.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 10 | `block_043.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 11 | `block_036.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 12 | `block_041.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 13 | `block_061.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 14 | `block_054.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 15 | `block_060.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.158 | candidate |
| 16 | `block_014.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 17 | `block_046.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 18 | `block_031.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 19 | `block_027.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 20 | `block_010.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 21 | `block_001.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 22 | `block_026.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 23 | `block_040.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 24 | `block_038.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 25 | `block_031.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 26 | `block_053.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 27 | `block_055.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 28 | `block_024.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 29 | `block_021.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 30 | `block_002.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 31 | `block_018.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 32 | `block_044.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 33 | `block_050.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 34 | `block_035.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 35 | `block_004.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 36 | `block_054.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 37 | `block_023.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 38 | `block_039.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 39 | `block_016.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 40 | `block_032.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 41 | `block_006.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 42 | `block_024.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 43 | `block_059.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 44 | `block_026.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 45 | `block_048.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 46 | `block_053.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 47 | `block_042.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 48 | `block_049.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 49 | `block_046.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 50 | `block_034.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 51 | `block_059.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 52 | `block_024.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 53 | `block_020.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 54 | `block_041.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 55 | `block_008.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 56 | `block_032.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 57 | `block_002.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 58 | `block_037.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 59 | `block_038.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 60 | `block_029.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 61 | `block_010.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 62 | `block_015.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 63 | `block_037.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.156 | candidate |
| 64 | `block_021.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 65 | `block_013.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 66 | `block_045.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 67 | `block_052.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 68 | `block_005.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 69 | `block_014.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 70 | `block_025.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 71 | `block_040.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.227 | 0.157 | candidate |
| 72 | `block_012.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 73 | `block_030.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 74 | `block_018.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 75 | `block_020.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 76 | `block_025.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 77 | `block_043.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 78 | `block_054.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 79 | `block_014.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 80 | `block_030.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 81 | `block_038.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 82 | `block_019.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 83 | `block_020.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 84 | `block_027.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 85 | `block_055.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 86 | `block_011.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 87 | `block_035.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.156 | candidate |
| 88 | `block_048.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 89 | `block_051.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 90 | `block_033.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 91 | `block_000.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 92 | `block_015.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 93 | `block_052.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 94 | `block_012.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 95 | `block_003.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 96 | `block_044.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 97 | `block_004.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 98 | `block_060.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 99 | `block_057.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 100 | `block_032.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 101 | `block_041.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 102 | `block_028.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 103 | `block_044.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 104 | `block_042.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 105 | `block_023.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 106 | `block_057.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 107 | `block_013.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 108 | `block_028.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 109 | `block_026.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 110 | `block_037.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 111 | `block_001.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 112 | `block_017.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 113 | `block_021.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 114 | `block_036.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 115 | `block_007.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 116 | `block_045.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 117 | `block_048.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 118 | `block_061.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 119 | `block_050.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 120 | `block_018.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 121 | `block_039.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 122 | `block_042.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 123 | `block_034.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 124 | `block_051.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 125 | `block_040.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 126 | `block_062.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 127 | `block_049.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 128 | `block_023.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 129 | `block_035.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 130 | `block_011.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 131 | `block_011.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 132 | `block_033.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 133 | `block_025.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 134 | `block_063.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 135 | `block_027.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 136 | `block_043.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 137 | `block_006.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 138 | `block_007.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 139 | `block_031.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 140 | `block_008.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 141 | `block_055.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 142 | `block_004.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 143 | `block_052.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 144 | `block_028.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 145 | `block_049.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 146 | `block_022.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 147 | `block_046.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 148 | `block_059.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 149 | `block_062.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 150 | `block_033.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 151 | `block_056.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 152 | `block_036.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 153 | `block_008.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 154 | `block_022.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 155 | `block_009.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 156 | `block_034.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 157 | `block_057.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 158 | `block_016.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 159 | `block_047.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 160 | `block_056.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 161 | `block_019.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 162 | `block_003.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 163 | `block_050.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 164 | `block_015.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 165 | `block_019.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 166 | `block_009.slot_01.moe_expert.down` | moe_expert.down | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 167 | `block_058.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 168 | `block_063.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 169 | `block_005.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 170 | `block_002.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 171 | `block_053.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 172 | `block_016.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 173 | `block_017.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 174 | `block_003.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 175 | `block_009.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 176 | `block_005.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 177 | `block_058.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 178 | `block_029.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.157 | candidate |
| 179 | `block_007.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 180 | `block_006.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 181 | `block_039.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 182 | `block_047.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 183 | `block_060.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 184 | `block_013.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 185 | `block_062.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 186 | `block_001.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 187 | `block_010.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 188 | `block_000.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 189 | `block_045.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 190 | `block_017.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 191 | `block_029.slot_02.moe_expert.up` | moe_expert.up | potential_target | 0.188 | 0.226 | 0.158 | candidate |
| 192 | `block_000.slot_00.moe_expert.gate` | moe_expert.gate | potential_target | 0.187 | 0.225 | 0.163 | candidate |
| 193 | `block_027.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 194 | `block_022.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 195 | `block_050.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 196 | `block_021.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 197 | `block_017.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 198 | `block_037.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 199 | `block_005.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 200 | `block_062.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 201 | `block_045.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 202 | `block_048.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 203 | `block_049.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 204 | `block_063.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 205 | `block_034.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 206 | `block_002.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 207 | `block_043.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 208 | `block_026.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 209 | `block_035.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 210 | `block_009.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 211 | `block_025.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 212 | `block_044.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 213 | `block_001.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 214 | `block_046.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 215 | `block_023.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 216 | `block_013.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 217 | `block_042.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 218 | `block_040.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 219 | `block_039.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 220 | `block_061.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 221 | `block_012.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 222 | `block_032.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.167 | 0.201 | 0.154 | candidate |
| 223 | `block_047.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 224 | `block_055.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 225 | `block_041.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 226 | `block_024.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 227 | `block_038.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.155 | candidate |
| 228 | `block_053.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.155 | candidate |
| 229 | `block_054.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.155 | candidate |
| 230 | `block_028.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 231 | `block_036.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 232 | `block_010.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 233 | `block_016.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 234 | `block_020.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 235 | `block_059.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.155 | candidate |
| 236 | `block_014.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 237 | `block_004.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 238 | `block_007.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 239 | `block_019.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 240 | `block_008.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 241 | `block_051.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 242 | `block_033.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.201 | 0.154 | candidate |
| 243 | `block_011.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.200 | 0.154 | candidate |
| 244 | `block_015.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.200 | 0.155 | candidate |
| 245 | `block_029.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.200 | 0.154 | candidate |
| 246 | `block_057.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.200 | 0.155 | candidate |
| 247 | `block_052.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.200 | 0.154 | candidate |
| 248 | `block_056.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.200 | 0.155 | candidate |
| 249 | `block_000.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.200 | 0.154 | candidate |
| 250 | `block_030.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.200 | 0.155 | candidate |
| 251 | `block_018.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.200 | 0.154 | candidate |
| 252 | `block_031.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.200 | 0.154 | candidate |
| 253 | `block_003.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.200 | 0.154 | candidate |
| 254 | `block_006.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.200 | 0.155 | candidate |
| 255 | `block_058.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.200 | 0.155 | candidate |
| 256 | `block_060.slot_04.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.200 | 0.155 | candidate |
| 257 | `block_024.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.166 | 0.199 | 0.157 | candidate |
| 258 | `block_018.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 259 | `block_031.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 260 | `block_042.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 261 | `block_035.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 262 | `block_016.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.157 | candidate |
| 263 | `block_056.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.157 | candidate |
| 264 | `block_060.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 265 | `block_036.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.157 | candidate |
| 266 | `block_062.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 267 | `block_053.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 268 | `block_020.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 269 | `block_029.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.157 | candidate |
| 270 | `block_002.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 271 | `block_044.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 272 | `block_011.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 273 | `block_049.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 274 | `block_046.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 275 | `block_032.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.157 | candidate |
| 276 | `block_057.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 277 | `block_039.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 278 | `block_006.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 279 | `block_028.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 280 | `block_038.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 281 | `block_037.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 282 | `block_047.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 283 | `block_004.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 284 | `block_026.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 285 | `block_022.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 286 | `block_027.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 287 | `block_019.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 288 | `block_061.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 289 | `block_052.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 290 | `block_041.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 291 | `block_017.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 292 | `block_033.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 293 | `block_063.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 294 | `block_045.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 295 | `block_055.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 296 | `block_000.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 297 | `block_008.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 298 | `block_025.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 299 | `block_007.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 300 | `block_054.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 301 | `block_015.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.158 | candidate |
| 302 | `block_058.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 303 | `block_059.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 304 | `block_034.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 305 | `block_048.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 306 | `block_021.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 307 | `block_003.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 308 | `block_050.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 309 | `block_023.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 310 | `block_001.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.161 | candidate |
| 311 | `block_014.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 312 | `block_013.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 313 | `block_005.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.199 | 0.159 | candidate |
| 314 | `block_051.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.198 | 0.160 | candidate |
| 315 | `block_040.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.198 | 0.159 | candidate |
| 316 | `block_043.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.165 | 0.198 | 0.160 | candidate |
| 317 | `block_010.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.164 | 0.198 | 0.161 | candidate |
| 318 | `block_012.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.164 | 0.198 | 0.161 | candidate |
| 319 | `block_030.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.164 | 0.198 | 0.162 | candidate |
| 320 | `block_009.slot_05.attn_proj_i8.model_width` | attn_proj_i8.model_width | potential_target | 0.164 | 0.198 | 0.161 | candidate |
| 321 | `block_004.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.197 | 0.159 | candidate |
| 322 | `block_008.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.197 | 0.158 | candidate |
| 323 | `block_035.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.197 | 0.158 | candidate |
| 324 | `block_063.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.197 | 0.158 | candidate |
| 325 | `block_013.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.197 | 0.158 | candidate |
| 326 | `block_016.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 327 | `block_056.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.157 | candidate |
| 328 | `block_002.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 329 | `block_062.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 330 | `block_062.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.157 | candidate |
| 331 | `block_010.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 332 | `block_054.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 333 | `block_034.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.157 | candidate |
| 334 | `block_000.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.157 | candidate |
| 335 | `block_061.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.157 | candidate |
| 336 | `block_035.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 337 | `block_020.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.157 | candidate |
| 338 | `block_015.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 339 | `block_032.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 340 | `block_045.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 341 | `block_023.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 342 | `block_011.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 343 | `block_029.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.157 | candidate |
| 344 | `block_055.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 345 | `block_033.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 346 | `block_040.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 347 | `block_001.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 348 | `block_037.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 349 | `block_031.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.157 | candidate |
| 350 | `block_012.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 351 | `block_022.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 352 | `block_042.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 353 | `block_057.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 354 | `block_017.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 355 | `block_044.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 356 | `block_047.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 357 | `block_017.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 358 | `block_045.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 359 | `block_041.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 360 | `block_053.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 361 | `block_002.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 362 | `block_056.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 363 | `block_007.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 364 | `block_003.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 365 | `block_021.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 366 | `block_049.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 367 | `block_061.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 368 | `block_063.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 369 | `block_048.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 370 | `block_014.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 371 | `block_019.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 372 | `block_036.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 373 | `block_014.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 374 | `block_039.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 375 | `block_040.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 376 | `block_049.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 377 | `block_039.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 378 | `block_027.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 379 | `block_004.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 380 | `block_019.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 381 | `block_005.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 382 | `block_059.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 383 | `block_025.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 384 | `block_038.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 385 | `block_003.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 386 | `block_028.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 387 | `block_024.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 388 | `block_059.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 389 | `block_009.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 390 | `block_025.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 391 | `block_013.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 392 | `block_041.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 393 | `block_031.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 394 | `block_018.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 395 | `block_046.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 396 | `block_018.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 397 | `block_027.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 398 | `block_006.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 399 | `block_022.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 400 | `block_029.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 401 | `block_026.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 402 | `block_047.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 403 | `block_037.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.158 | candidate |
| 404 | `block_016.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 405 | `block_057.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 406 | `block_055.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 407 | `block_042.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 408 | `block_046.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 409 | `block_052.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 410 | `block_024.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 411 | `block_005.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 412 | `block_001.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.160 | candidate |
| 413 | `block_032.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 414 | `block_038.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.163 | 0.196 | 0.159 | candidate |
| 415 | `block_060.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.160 | candidate |
| 416 | `block_028.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.159 | candidate |
| 417 | `block_015.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.159 | candidate |
| 418 | `block_023.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.159 | candidate |
| 419 | `block_000.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.158 | candidate |
| 420 | `block_012.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.160 | candidate |
| 421 | `block_020.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.159 | candidate |
| 422 | `block_026.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.159 | candidate |
| 423 | `block_021.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.159 | candidate |
| 424 | `block_048.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.159 | candidate |
| 425 | `block_011.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.160 | candidate |
| 426 | `block_052.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.159 | candidate |
| 427 | `block_034.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.159 | candidate |
| 428 | `block_044.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.160 | candidate |
| 429 | `block_036.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.159 | candidate |
| 430 | `block_060.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.159 | candidate |
| 431 | `block_058.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.160 | candidate |
| 432 | `block_058.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.159 | candidate |
| 433 | `block_008.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.160 | candidate |
| 434 | `block_007.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.159 | candidate |
| 435 | `block_043.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.159 | candidate |
| 436 | `block_033.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.160 | candidate |
| 437 | `block_043.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.160 | candidate |
| 438 | `block_010.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.160 | candidate |
| 439 | `block_030.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.160 | candidate |
| 440 | `block_006.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.160 | candidate |
| 441 | `block_051.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.196 | 0.160 | candidate |
| 442 | `block_050.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.195 | 0.160 | candidate |
| 443 | `block_009.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.195 | 0.160 | candidate |
| 444 | `block_050.slot_06.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.195 | 0.160 | candidate |
| 445 | `block_051.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.195 | 0.164 | candidate |
| 446 | `block_030.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.195 | 0.162 | candidate |
| 447 | `block_053.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.195 | 0.162 | candidate |
| 448 | `block_054.slot_03.attn_proj_i8.narrow` | attn_proj_i8.narrow | potential_target | 0.162 | 0.195 | 0.161 | candidate |

## Routing-critical tensors

| Tensor | Readiness | Risk | Reasons |
| ------ | --------: | ---: | ------- |
| `block_060.slot_11.router` | 0.056 | 0.651 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure |
| `block_008.slot_11.router` | 0.054 | 0.682 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.2503<br>outlier_fraction=0.0005<br>peak_to_rms=9.185<br>linked to routing structure |
| `block_006.slot_11.router` | 0.054 | 0.671 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.1552<br>outlier_fraction=0.0003<br>peak_to_rms=7.640<br>linked to routing structure |
| `block_028.slot_11.router` | 0.054 | 0.686 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.2493<br>outlier_fraction=0.0003<br>peak_to_rms=9.826<br>linked to routing structure |
| `block_058.slot_11.router` | 0.054 | 0.669 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.1163<br>outlier_fraction=0.0000<br>peak_to_rms=7.371<br>linked to routing structure |
| `block_049.slot_11.router` | 0.054 | 0.672 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.1313<br>outlier_fraction=0.0000<br>peak_to_rms=7.844<br>linked to routing structure |
| `block_000.slot_11.router` | 0.053 | 0.662 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0375<br>outlier_fraction=0.0000<br>peak_to_rms=6.397<br>linked to routing structure |
| `block_054.slot_11.router` | 0.053 | 0.668 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0407<br>outlier_fraction=0.0001<br>peak_to_rms=7.192<br>linked to routing structure |
| `block_057.slot_11.router` | 0.053 | 0.676 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0904<br>outlier_fraction=0.0001<br>peak_to_rms=8.290<br>linked to routing structure |
| `block_043.slot_11.router` | 0.053 | 0.672 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0603<br>outlier_fraction=0.0001<br>peak_to_rms=7.809<br>linked to routing structure |
| `block_038.slot_11.router` | 0.053 | 0.688 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.1741<br>outlier_fraction=0.0005<br>peak_to_rms=10.042<br>linked to routing structure |
| `block_001.slot_11.router` | 0.052 | 0.671 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0288<br>outlier_fraction=0.0001<br>peak_to_rms=7.635<br>linked to routing structure |
| `block_052.slot_11.router` | 0.052 | 0.672 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0318<br>outlier_fraction=0.0001<br>peak_to_rms=7.742<br>linked to routing structure |
| `block_042.slot_11.router` | 0.052 | 0.684 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.1140<br>outlier_fraction=0.0002<br>peak_to_rms=9.499<br>linked to routing structure |
| `block_053.slot_11.router` | 0.052 | 0.678 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0666<br>outlier_fraction=0.0001<br>peak_to_rms=8.615<br>linked to routing structure |
| `block_017.slot_11.router` | 0.052 | 0.684 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.1137<br>outlier_fraction=0.0002<br>peak_to_rms=9.515<br>linked to routing structure |
| `block_059.slot_11.router` | 0.052 | 0.673 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0286<br>outlier_fraction=0.0000<br>peak_to_rms=7.920<br>linked to routing structure |
| `block_030.slot_11.router` | 0.052 | 0.674 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0323<br>outlier_fraction=0.0004<br>peak_to_rms=8.104<br>linked to routing structure |
| `block_051.slot_11.router` | 0.052 | 0.685 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0943<br>outlier_fraction=0.0001<br>peak_to_rms=9.627<br>linked to routing structure |
| `block_035.slot_11.router` | 0.052 | 0.679 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0531<br>outlier_fraction=0.0004<br>peak_to_rms=8.856<br>linked to routing structure |
| `block_013.slot_11.router` | 0.052 | 0.676 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0201<br>outlier_fraction=0.0005<br>peak_to_rms=8.308<br>linked to routing structure |
| `block_022.slot_11.router` | 0.052 | 0.679 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0437<br>outlier_fraction=0.0004<br>peak_to_rms=8.760<br>linked to routing structure |
| `block_048.slot_11.router` | 0.052 | 0.677 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0208<br>outlier_fraction=0.0001<br>peak_to_rms=8.544<br>linked to routing structure |
| `block_024.slot_11.router` | 0.051 | 0.684 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0662<br>outlier_fraction=0.0004<br>peak_to_rms=9.469<br>linked to routing structure |
| `block_021.slot_11.router` | 0.051 | 0.680 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0357<br>outlier_fraction=0.0002<br>peak_to_rms=8.956<br>linked to routing structure |
| `block_055.slot_11.router` | 0.051 | 0.685 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0671<br>outlier_fraction=0.0003<br>peak_to_rms=9.581<br>linked to routing structure |
| `block_018.slot_11.router` | 0.051 | 0.680 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0292<br>outlier_fraction=0.0002<br>peak_to_rms=8.976<br>linked to routing structure |
| `block_026.slot_11.router` | 0.051 | 0.682 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0397<br>outlier_fraction=0.0004<br>peak_to_rms=9.271<br>linked to routing structure |
| `block_019.slot_11.router` | 0.051 | 0.691 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.1021<br>outlier_fraction=0.0005<br>peak_to_rms=10.447<br>linked to routing structure |
| `block_010.slot_11.router` | 0.051 | 0.681 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0263<br>outlier_fraction=0.0003<br>peak_to_rms=9.071<br>linked to routing structure |
| `block_041.slot_11.router` | 0.051 | 0.689 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0829<br>outlier_fraction=0.0004<br>peak_to_rms=10.177<br>linked to routing structure |
| `block_016.slot_11.router` | 0.051 | 0.683 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0254<br>outlier_fraction=0.0002<br>peak_to_rms=9.308<br>linked to routing structure |
| `block_014.slot_11.router` | 0.051 | 0.683 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0238<br>outlier_fraction=0.0002<br>peak_to_rms=9.386<br>linked to routing structure |
| `block_050.slot_11.router` | 0.051 | 0.691 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0802<br>outlier_fraction=0.0001<br>peak_to_rms=10.447<br>linked to routing structure |
| `block_039.slot_11.router` | 0.051 | 0.685 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0304<br>outlier_fraction=0.0005<br>peak_to_rms=9.628<br>linked to routing structure |
| `block_046.slot_11.router` | 0.051 | 0.686 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0357<br>outlier_fraction=0.0002<br>peak_to_rms=9.741<br>linked to routing structure |
| `block_037.slot_11.router` | 0.051 | 0.685 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0286<br>outlier_fraction=0.0003<br>peak_to_rms=9.657<br>linked to routing structure |
| `block_036.slot_11.router` | 0.051 | 0.687 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0458<br>outlier_fraction=0.0003<br>peak_to_rms=9.987<br>linked to routing structure |
| `block_015.slot_11.router` | 0.051 | 0.687 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0403<br>outlier_fraction=0.0003<br>peak_to_rms=9.930<br>linked to routing structure |
| `block_004.slot_11.router` | 0.051 | 0.687 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0352<br>outlier_fraction=0.0005<br>peak_to_rms=9.895<br>linked to routing structure |
| `block_025.slot_11.router` | 0.051 | 0.686 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0262<br>outlier_fraction=0.0003<br>peak_to_rms=9.755<br>linked to routing structure |
| `block_040.slot_11.router` | 0.050 | 0.688 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0311<br>outlier_fraction=0.0005<br>peak_to_rms=10.133<br>linked to routing structure |
| `block_027.slot_11.router` | 0.050 | 0.690 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0341<br>outlier_fraction=0.0003<br>peak_to_rms=10.292<br>linked to routing structure |
| `block_009.slot_11.router` | 0.050 | 0.690 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0251<br>outlier_fraction=0.0007<br>peak_to_rms=10.396<br>linked to routing structure |
| `block_045.slot_11.router` | 0.050 | 0.692 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0338<br>outlier_fraction=0.0009<br>peak_to_rms=10.621<br>linked to routing structure |
| `block_012.slot_11.router` | 0.050 | 0.695 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0314<br>outlier_fraction=0.0004<br>peak_to_rms=11.045<br>linked to routing structure |
| `block_056.slot_11.router` | 0.050 | 0.697 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0443<br>outlier_fraction=0.0008<br>peak_to_rms=11.349<br>linked to routing structure |
| `block_020.slot_11.router` | 0.050 | 0.694 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0221<br>outlier_fraction=0.0008<br>peak_to_rms=10.983<br>linked to routing structure |
| `block_007.slot_11.router` | 0.050 | 0.710 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.1395<br>outlier_fraction=0.0005<br>peak_to_rms=13.223<br>linked to routing structure |
| `block_023.slot_11.router` | 0.049 | 0.699 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0381<br>outlier_fraction=0.0009<br>peak_to_rms=11.652<br>linked to routing structure |
| `block_029.slot_11.router` | 0.049 | 0.699 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0314<br>outlier_fraction=0.0006<br>peak_to_rms=11.586<br>linked to routing structure |
| `block_032.slot_11.router` | 0.049 | 0.699 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0225<br>outlier_fraction=0.0003<br>peak_to_rms=11.673<br>linked to routing structure |
| `block_044.slot_11.router` | 0.049 | 0.704 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0528<br>outlier_fraction=0.0002<br>peak_to_rms=12.375<br>linked to routing structure |
| `block_002.slot_11.router` | 0.049 | 0.701 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0247<br>outlier_fraction=0.0005<br>peak_to_rms=11.872<br>linked to routing structure |
| `block_034.slot_11.router` | 0.049 | 0.704 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0466<br>outlier_fraction=0.0015<br>peak_to_rms=12.307<br>linked to routing structure |
| `block_061.slot_11.router` | 0.049 | 0.704 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0243<br>outlier_fraction=0.0013<br>peak_to_rms=12.314<br>linked to routing structure |
| `block_003.slot_11.router` | 0.049 | 0.706 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0388<br>outlier_fraction=0.0013<br>peak_to_rms=12.607<br>linked to routing structure |
| `block_005.slot_11.router` | 0.048 | 0.709 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0224<br>outlier_fraction=0.0007<br>peak_to_rms=13.139<br>linked to routing structure |
| `block_031.slot_11.router` | 0.048 | 0.712 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0335<br>outlier_fraction=0.0006<br>peak_to_rms=13.529<br>linked to routing structure |
| `block_047.slot_11.router` | 0.048 | 0.715 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0462<br>outlier_fraction=0.0003<br>peak_to_rms=13.935<br>linked to routing structure |
| `block_062.slot_11.router` | 0.047 | 0.717 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0208<br>outlier_fraction=0.0011<br>peak_to_rms=14.233<br>linked to routing structure |
| `block_063.slot_11.router` | 0.046 | 0.727 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0470<br>outlier_fraction=0.0009<br>peak_to_rms=15.632<br>linked to routing structure |
| `block_033.slot_11.router` | 0.046 | 0.726 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0300<br>outlier_fraction=0.0005<br>peak_to_rms=15.542<br>linked to routing structure |
| `block_011.slot_11.router` | 0.046 | 0.727 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0238<br>outlier_fraction=0.0005<br>peak_to_rms=15.616<br>linked to routing structure |

## Precision-sensitive tensors

| Tensor | Risk | Reasons |
| ------ | ---: | ------- |
| `block_042.slot_08.block_norm` | 0.561 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0010<br>peak_to_rms=1.260<br>lives in a routing-critical block |
| `block_023.slot_09.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0094<br>outlier_fraction=0.0000<br>peak_to_rms=1.466<br>lives in a routing-critical block |
| `block_012.slot_09.block_norm` | 0.565 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0200<br>outlier_fraction=0.0000<br>peak_to_rms=1.738<br>lives in a routing-critical block |
| `block_061.slot_09.block_norm` | 0.564 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0098<br>outlier_fraction=0.0000<br>peak_to_rms=1.631<br>lives in a routing-critical block |
| `block_009.slot_09.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0007<br>outlier_fraction=0.0000<br>peak_to_rms=1.464<br>lives in a routing-critical block |
| `block_010.slot_09.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0008<br>outlier_fraction=0.0000<br>peak_to_rms=1.478<br>lives in a routing-critical block |
| `block_034.slot_09.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0041<br>outlier_fraction=0.0000<br>peak_to_rms=1.546<br>lives in a routing-critical block |
| `block_047.slot_08.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0002<br>peak_to_rms=1.476<br>lives in a routing-critical block |
| `block_004.slot_09.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0018<br>outlier_fraction=0.0000<br>peak_to_rms=1.532<br>lives in a routing-critical block |
| `block_003.slot_09.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0008<br>outlier_fraction=0.0000<br>peak_to_rms=1.515<br>lives in a routing-critical block |
| `block_045.slot_09.block_norm` | 0.564 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0042<br>outlier_fraction=0.0000<br>peak_to_rms=1.581<br>lives in a routing-critical block |
| `block_002.slot_09.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0013<br>outlier_fraction=0.0000<br>peak_to_rms=1.534<br>lives in a routing-critical block |
| `block_008.slot_09.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0008<br>outlier_fraction=0.0000<br>peak_to_rms=1.530<br>lives in a routing-critical block |
| `block_010.slot_07.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.518<br>lives in a routing-critical block |
| `block_007.slot_09.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0007<br>outlier_fraction=0.0000<br>peak_to_rms=1.532<br>lives in a routing-critical block |
| `block_011.slot_07.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.520<br>lives in a routing-critical block |
| `block_005.slot_09.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0013<br>outlier_fraction=0.0000<br>peak_to_rms=1.548<br>lives in a routing-critical block |
| `block_063.slot_09.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0015<br>outlier_fraction=0.0000<br>peak_to_rms=1.560<br>lives in a routing-critical block |
| `block_006.slot_09.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0011<br>outlier_fraction=0.0000<br>peak_to_rms=1.560<br>lives in a routing-critical block |
| `block_062.slot_09.block_norm` | 0.564 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0033<br>outlier_fraction=0.0000<br>peak_to_rms=1.616<br>lives in a routing-critical block |
| `block_008.slot_07.block_norm` | 0.563 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.570<br>lives in a routing-critical block |
| `block_006.slot_07.block_norm` | 0.564 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.578<br>lives in a routing-critical block |
| `block_004.slot_07.block_norm` | 0.564 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.580<br>lives in a routing-critical block |
| `block_005.slot_07.block_norm` | 0.564 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.587<br>lives in a routing-critical block |
| `block_056.slot_09.block_norm` | 0.564 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0024<br>outlier_fraction=0.0000<br>peak_to_rms=1.637<br>lives in a routing-critical block |
| `block_003.slot_07.block_norm` | 0.564 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.603<br>lives in a routing-critical block |
| `block_007.slot_07.block_norm` | 0.564 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.622<br>lives in a routing-critical block |
| `block_009.slot_07.block_norm` | 0.564 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.628<br>lives in a routing-critical block |
| `block_062.slot_07.block_norm` | 0.564 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0005<br>outlier_fraction=0.0000<br>peak_to_rms=1.644<br>lives in a routing-critical block |
| `block_046.slot_08.block_norm` | 0.564 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0005<br>peak_to_rms=1.638<br>lives in a routing-critical block |
| `block_063.slot_07.block_norm` | 0.564 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0002<br>outlier_fraction=0.0000<br>peak_to_rms=1.657<br>lives in a routing-critical block |
| `block_002.slot_07.block_norm` | 0.565 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.722<br>lives in a routing-critical block |
| `block_005.slot_10.block_norm` | 0.565 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0010<br>peak_to_rms=1.758<br>lives in a routing-critical block |
| `block_011.slot_09.block_norm` | 0.565 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0013<br>outlier_fraction=0.0018<br>peak_to_rms=1.419<br>lives in a routing-critical block |
| `block_023.slot_07.block_norm` | 0.565 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.774<br>lives in a routing-critical block |
| `block_028.slot_10.block_norm` | 0.565 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0016<br>peak_to_rms=1.775<br>lives in a routing-critical block |
| `block_013.slot_07.block_norm` | 0.565 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0005<br>peak_to_rms=1.782<br>lives in a routing-critical block |
| `block_034.slot_07.block_norm` | 0.565 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.801<br>lives in a routing-critical block |
| `block_056.slot_07.block_norm` | 0.565 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.811<br>lives in a routing-critical block |
| `block_045.slot_07.block_norm` | 0.565 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.811<br>lives in a routing-critical block |
| `block_012.slot_07.block_norm` | 0.565 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.812<br>lives in a routing-critical block |
| `block_055.slot_09.block_norm` | 0.565 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0005<br>outlier_fraction=0.0015<br>peak_to_rms=1.845<br>lives in a routing-critical block |
| `block_024.slot_10.block_norm` | 0.565 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0013<br>peak_to_rms=1.846<br>lives in a routing-critical block |
| `block_014.slot_07.block_norm` | 0.566 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0010<br>peak_to_rms=1.871<br>lives in a routing-critical block |
| `block_018.slot_07.block_norm` | 0.566 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0008<br>peak_to_rms=1.878<br>lives in a routing-critical block |
| `block_007.slot_10.block_norm` | 0.566 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0007<br>peak_to_rms=1.897<br>lives in a routing-critical block |
| `block_016.slot_07.block_norm` | 0.566 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0013<br>peak_to_rms=1.912<br>lives in a routing-critical block |
| `block_054.slot_08.block_norm` | 0.566 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.915<br>lives in a routing-critical block |
| `block_061.slot_07.block_norm` | 0.566 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0000<br>peak_to_rms=1.922<br>lives in a routing-critical block |
| `block_013.slot_10.block_norm` | 0.566 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0010<br>peak_to_rms=1.928<br>lives in a routing-critical block |
| `block_013.slot_09.block_norm` | 0.566 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0011<br>outlier_fraction=0.0020<br>peak_to_rms=1.416<br>lives in a routing-critical block |
| `block_019.slot_07.block_norm` | 0.566 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0011<br>peak_to_rms=1.939<br>lives in a routing-critical block |
| `block_025.slot_10.block_norm` | 0.566 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0020<br>peak_to_rms=1.729<br>lives in a routing-critical block |
| `block_015.slot_07.block_norm` | 0.566 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0018<br>peak_to_rms=1.983<br>lives in a routing-critical block |
| `block_029.slot_10.block_norm` | 0.567 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0020<br>peak_to_rms=2.032<br>lives in a routing-critical block |
| `block_058.slot_08.block_norm` | 0.567 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0010<br>peak_to_rms=2.034<br>lives in a routing-critical block |
| `block_026.slot_10.block_norm` | 0.567 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0015<br>peak_to_rms=2.057<br>lives in a routing-critical block |
| `block_018.slot_10.block_norm` | 0.567 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0015<br>peak_to_rms=2.107<br>lives in a routing-critical block |
| `block_036.slot_10.block_norm` | 0.567 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0021<br>peak_to_rms=1.777<br>lives in a routing-critical block |
| `block_037.slot_10.block_norm` | 0.567 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0021<br>peak_to_rms=1.509<br>lives in a routing-critical block |
| `block_016.slot_10.block_norm` | 0.568 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0010<br>peak_to_rms=2.154<br>lives in a routing-critical block |
| `block_031.slot_10.block_norm` | 0.568 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0020<br>peak_to_rms=2.221<br>lives in a routing-critical block |
| `block_057.slot_09.block_norm` | 0.568 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0002<br>outlier_fraction=0.0023<br>peak_to_rms=1.919<br>lives in a routing-critical block |
| `block_035.slot_10.block_norm` | 0.568 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0023<br>peak_to_rms=1.632<br>lives in a routing-critical block |
| `block_014.slot_10.block_norm` | 0.569 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0010<br>peak_to_rms=2.288<br>lives in a routing-critical block |
| `block_060.slot_08.block_norm` | 0.569 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0003<br>peak_to_rms=2.305<br>lives in a routing-critical block |
| `block_001.slot_07.block_norm` | 0.569 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0005<br>peak_to_rms=2.348<br>lives in a routing-critical block |
| `block_027.slot_10.block_norm` | 0.569 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0015<br>peak_to_rms=2.359<br>lives in a routing-critical block |
| `block_053.slot_08.block_norm` | 0.569 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0008<br>peak_to_rms=2.419<br>lives in a routing-critical block |
| `block_050.slot_09.block_norm` | 0.570 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0005<br>outlier_fraction=0.0024<br>peak_to_rms=1.633<br>lives in a routing-critical block |
| `block_054.slot_09.block_norm` | 0.570 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0005<br>outlier_fraction=0.0024<br>peak_to_rms=1.566<br>lives in a routing-critical block |
| `final_norm.slot_00.final_norm` | 0.570 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0024<br>peak_to_rms=2.157 |
| `block_033.slot_10.block_norm` | 0.570 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0024<br>peak_to_rms=2.086<br>lives in a routing-critical block |
| `block_041.slot_10.block_norm` | 0.570 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0024<br>peak_to_rms=1.662<br>lives in a routing-critical block |
| `block_022.slot_10.block_norm` | 0.570 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0018<br>peak_to_rms=2.458<br>lives in a routing-critical block |
| `block_050.slot_08.block_norm` | 0.570 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0005<br>peak_to_rms=2.471<br>lives in a routing-critical block |
| `block_020.slot_10.block_norm` | 0.570 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0016<br>peak_to_rms=2.484<br>lives in a routing-critical block |
| `block_051.slot_09.block_norm` | 0.571 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0003<br>outlier_fraction=0.0026<br>peak_to_rms=1.668<br>lives in a routing-critical block |
| `block_053.slot_09.block_norm` | 0.571 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0003<br>outlier_fraction=0.0026<br>peak_to_rms=1.588<br>lives in a routing-critical block |
| `block_032.slot_10.block_norm` | 0.571 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0026<br>peak_to_rms=1.828<br>lives in a routing-critical block |
| `block_059.slot_08.block_norm` | 0.571 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0011<br>peak_to_rms=2.701<br>lives in a routing-critical block |
| `block_014.slot_09.block_norm` | 0.572 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0015<br>outlier_fraction=0.0028<br>peak_to_rms=1.388<br>lives in a routing-critical block |
| `block_052.slot_09.block_norm` | 0.572 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0005<br>outlier_fraction=0.0028<br>peak_to_rms=1.591<br>lives in a routing-critical block |
| `block_049.slot_08.block_norm` | 0.572 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0020<br>peak_to_rms=2.766<br>lives in a routing-critical block |
| `block_017.slot_07.block_norm` | 0.572 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0028<br>peak_to_rms=2.563<br>lives in a routing-critical block |
| `block_017.slot_08.block_norm` | 0.572 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0028<br>peak_to_rms=2.680<br>lives in a routing-critical block |
| `block_038.slot_10.block_norm` | 0.572 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0028<br>peak_to_rms=1.640<br>lives in a routing-critical block |
| `block_039.slot_08.block_norm` | 0.572 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0028<br>peak_to_rms=1.311<br>lives in a routing-critical block |
| `block_058.slot_09.block_norm` | 0.572 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0028<br>peak_to_rms=2.363<br>lives in a routing-critical block |
| `block_021.slot_10.block_norm` | 0.572 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0013<br>peak_to_rms=2.835<br>lives in a routing-critical block |
| `block_059.slot_10.block_norm` | 0.573 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0023<br>peak_to_rms=2.895<br>lives in a routing-critical block |
| `block_014.slot_08.block_norm` | 0.573 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0024<br>peak_to_rms=2.896<br>lives in a routing-critical block |
| `block_019.slot_08.block_norm` | 0.573 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0029<br>peak_to_rms=1.987<br>lives in a routing-critical block |
| `block_037.slot_08.block_norm` | 0.573 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0015<br>peak_to_rms=2.978<br>lives in a routing-critical block |
| `block_013.slot_08.block_norm` | 0.573 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0020<br>peak_to_rms=2.979<br>lives in a routing-critical block |
| `block_019.slot_10.block_norm` | 0.573 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0016<br>peak_to_rms=2.989<br>lives in a routing-critical block |
| `block_001.slot_09.block_norm` | 0.576 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0218<br>outlier_fraction=0.0002<br>peak_to_rms=3.420<br>lives in a routing-critical block |
| `block_030.slot_10.block_norm` | 0.574 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0016<br>peak_to_rms=3.074<br>lives in a routing-critical block |
| `block_057.slot_08.block_norm` | 0.574 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0007<br>peak_to_rms=3.077<br>lives in a routing-critical block |
| `block_018.slot_08.block_norm` | 0.574 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0031<br>peak_to_rms=2.951<br>lives in a routing-critical block |
| `block_044.slot_08.block_norm` | 0.574 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0031<br>peak_to_rms=2.217<br>lives in a routing-critical block |
| `block_052.slot_08.block_norm` | 0.574 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0008<br>peak_to_rms=3.109<br>lives in a routing-critical block |
| `block_043.slot_08.block_norm` | 0.575 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0031<br>peak_to_rms=3.161<br>lives in a routing-critical block |
| `block_030.slot_08.block_norm` | 0.575 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0011<br>peak_to_rms=3.185<br>lives in a routing-critical block |
| `block_015.slot_08.block_norm` | 0.575 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0020<br>peak_to_rms=3.224<br>lives in a routing-critical block |
| `block_049.slot_09.block_norm` | 0.575 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0007<br>outlier_fraction=0.0033<br>peak_to_rms=1.599<br>lives in a routing-critical block |
| `block_029.slot_08.block_norm` | 0.575 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0033<br>peak_to_rms=2.097<br>lives in a routing-critical block |
| `block_033.slot_08.block_norm` | 0.575 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0033<br>peak_to_rms=2.912<br>lives in a routing-critical block |
| `block_040.slot_10.block_norm` | 0.575 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0033<br>peak_to_rms=1.536<br>lives in a routing-critical block |
| `block_043.slot_10.block_norm` | 0.575 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0033<br>peak_to_rms=2.089<br>lives in a routing-critical block |
| `block_028.slot_08.block_norm` | 0.576 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0034<br>peak_to_rms=1.411<br>lives in a routing-critical block |
| `block_042.slot_10.block_norm` | 0.576 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0034<br>peak_to_rms=1.367<br>lives in a routing-critical block |
| `block_055.slot_08.block_norm` | 0.576 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0021<br>peak_to_rms=3.426<br>lives in a routing-critical block |
| `block_022.slot_08.block_norm` | 0.577 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0021<br>peak_to_rms=3.435<br>lives in a routing-critical block |
| `block_011.slot_08.block_norm` | 0.577 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0011<br>peak_to_rms=3.550<br>lives in a routing-critical block |
| `block_029.slot_07.block_norm` | 0.578 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0036<br>peak_to_rms=2.704<br>lives in a routing-critical block |
| `block_038.slot_08.block_norm` | 0.578 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0036<br>peak_to_rms=1.386<br>lives in a routing-critical block |
| `block_039.slot_10.block_norm` | 0.578 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0036<br>peak_to_rms=1.381<br>lives in a routing-critical block |
| `block_010.slot_08.block_norm` | 0.578 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0008<br>peak_to_rms=3.670<br>lives in a routing-critical block |
| `block_060.slot_10.block_norm` | 0.579 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0018<br>peak_to_rms=3.726<br>lives in a routing-critical block |
| `block_047.slot_09.block_norm` | 0.579 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0008<br>outlier_fraction=0.0037<br>peak_to_rms=1.582<br>lives in a routing-critical block |
| `block_048.slot_08.block_norm` | 0.579 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0026<br>peak_to_rms=3.853<br>lives in a routing-critical block |
| `block_048.slot_09.block_norm` | 0.580 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0011<br>outlier_fraction=0.0039<br>peak_to_rms=1.595<br>lives in a routing-critical block |
| `block_031.slot_08.block_norm` | 0.580 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0039<br>peak_to_rms=3.367<br>lives in a routing-critical block |
| `block_041.slot_08.block_norm` | 0.580 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0039<br>peak_to_rms=1.588<br>lives in a routing-critical block |
| `block_044.slot_10.block_norm` | 0.580 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0039<br>peak_to_rms=1.372<br>lives in a routing-critical block |
| `block_051.slot_08.block_norm` | 0.581 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0002<br>peak_to_rms=4.016<br>lives in a routing-critical block |
| `block_020.slot_07.block_norm` | 0.581 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0041<br>peak_to_rms=2.796<br>lives in a routing-critical block |
| `block_026.slot_08.block_norm` | 0.581 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0041<br>peak_to_rms=3.696<br>lives in a routing-critical block |
| `block_009.slot_08.block_norm` | 0.582 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0002<br>peak_to_rms=4.143<br>lives in a routing-critical block |
| `block_007.slot_08.block_norm` | 0.582 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0003<br>peak_to_rms=4.214<br>lives in a routing-critical block |
| `block_060.slot_07.block_norm` | 0.582 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0042<br>peak_to_rms=3.432<br>lives in a routing-critical block |
| `block_050.slot_07.block_norm` | 0.583 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0039<br>peak_to_rms=4.385<br>lives in a routing-critical block |
| `block_016.slot_08.block_norm` | 0.583 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0044<br>peak_to_rms=1.800<br>lives in a routing-critical block |
| `block_022.slot_07.block_norm` | 0.583 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0044<br>peak_to_rms=2.651<br>lives in a routing-critical block |
| `block_024.slot_08.block_norm` | 0.583 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0044<br>peak_to_rms=1.357<br>lives in a routing-critical block |
| `block_027.slot_08.block_norm` | 0.583 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0044<br>peak_to_rms=2.310<br>lives in a routing-critical block |
| `block_030.slot_07.block_norm` | 0.583 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0044<br>peak_to_rms=3.372<br>lives in a routing-critical block |
| `block_008.slot_08.block_norm` | 0.584 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0007<br>peak_to_rms=4.466<br>lives in a routing-critical block |
| `block_024.slot_07.block_norm` | 0.584 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0046<br>peak_to_rms=2.442<br>lives in a routing-critical block |
| `block_032.slot_08.block_norm` | 0.584 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0046<br>peak_to_rms=2.163<br>lives in a routing-critical block |
| `block_006.slot_08.block_norm` | 0.584 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0031<br>peak_to_rms=4.563<br>lives in a routing-critical block |
| `block_009.slot_10.block_norm` | 0.585 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0010<br>peak_to_rms=4.609<br>lives in a routing-critical block |
| `block_053.slot_07.block_norm` | 0.585 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0046<br>peak_to_rms=4.610<br>lives in a routing-critical block |
| `block_044.slot_09.block_norm` | 0.586 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0013<br>outlier_fraction=0.0047<br>peak_to_rms=1.539<br>lives in a routing-critical block |
| `block_020.slot_08.block_norm` | 0.586 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0047<br>peak_to_rms=1.759<br>lives in a routing-critical block |
| `block_036.slot_08.block_norm` | 0.586 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0047<br>peak_to_rms=1.439<br>lives in a routing-critical block |
| `block_049.slot_10.block_norm` | 0.586 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0047<br>peak_to_rms=1.339<br>lives in a routing-critical block |
| `block_046.slot_09.block_norm` | 0.587 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0013<br>outlier_fraction=0.0049<br>peak_to_rms=1.552<br>lives in a routing-critical block |
| `block_031.slot_07.block_norm` | 0.587 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0049<br>peak_to_rms=4.283<br>lives in a routing-critical block |
| `block_037.slot_07.block_norm` | 0.587 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0049<br>peak_to_rms=3.750<br>lives in a routing-critical block |
| `block_046.slot_10.block_norm` | 0.587 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0049<br>peak_to_rms=1.354<br>lives in a routing-critical block |
| `block_025.slot_08.block_norm` | 0.588 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0050<br>peak_to_rms=3.844<br>lives in a routing-critical block |
| `block_035.slot_08.block_norm` | 0.588 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0050<br>peak_to_rms=4.716<br>lives in a routing-critical block |
| `block_059.slot_07.block_norm` | 0.588 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0050<br>peak_to_rms=4.214<br>lives in a routing-critical block |
| `block_059.slot_09.block_norm` | 0.588 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0050<br>peak_to_rms=3.086<br>lives in a routing-critical block |
| `block_060.slot_09.block_norm` | 0.588 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0050<br>peak_to_rms=4.057<br>lives in a routing-critical block |
| `block_017.slot_10.block_norm` | 0.588 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0013<br>peak_to_rms=5.090<br>lives in a routing-critical block |
| `block_043.slot_09.block_norm` | 0.589 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0013<br>outlier_fraction=0.0052<br>peak_to_rms=1.557<br>lives in a routing-critical block |
| `block_015.slot_09.block_norm` | 0.589 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0007<br>outlier_fraction=0.0052<br>peak_to_rms=1.509<br>lives in a routing-critical block |
| `block_021.slot_08.block_norm` | 0.589 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0052<br>peak_to_rms=3.238<br>lives in a routing-critical block |
| `block_050.slot_10.block_norm` | 0.589 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0052<br>peak_to_rms=1.284<br>lives in a routing-critical block |
| `block_052.slot_07.block_norm` | 0.589 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0046<br>peak_to_rms=5.269<br>lives in a routing-critical block |
| `block_042.slot_09.block_norm` | 0.590 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0011<br>outlier_fraction=0.0054<br>peak_to_rms=1.533<br>lives in a routing-critical block |
| `block_040.slot_09.block_norm` | 0.591 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0013<br>outlier_fraction=0.0055<br>peak_to_rms=1.501<br>lives in a routing-critical block |
| `block_041.slot_09.block_norm` | 0.591 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0013<br>outlier_fraction=0.0055<br>peak_to_rms=1.556<br>lives in a routing-critical block |
| `block_048.slot_10.block_norm` | 0.591 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0002<br>outlier_fraction=0.0055<br>peak_to_rms=2.100<br>lives in a routing-critical block |
| `block_039.slot_07.block_norm` | 0.591 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0055<br>peak_to_rms=3.964<br>lives in a routing-critical block |
| `block_048.slot_07.block_norm` | 0.591 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0055<br>peak_to_rms=4.494<br>lives in a routing-critical block |
| `block_058.slot_10.block_norm` | 0.591 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0055<br>peak_to_rms=1.580<br>lives in a routing-critical block |
| `block_005.slot_08.block_norm` | 0.592 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0028<br>peak_to_rms=5.606<br>lives in a routing-critical block |
| `block_043.slot_07.block_norm` | 0.592 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0041<br>peak_to_rms=5.607<br>lives in a routing-critical block |
| `block_038.slot_09.block_norm` | 0.592 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0010<br>outlier_fraction=0.0057<br>peak_to_rms=1.491<br>lives in a routing-critical block |
| `block_025.slot_07.block_norm` | 0.592 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0057<br>peak_to_rms=3.327<br>lives in a routing-critical block |
| `block_027.slot_07.block_norm` | 0.592 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0057<br>peak_to_rms=2.530<br>lives in a routing-critical block |
| `block_038.slot_07.block_norm` | 0.592 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0057<br>peak_to_rms=4.345<br>lives in a routing-critical block |
| `block_046.slot_07.block_norm` | 0.592 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0057<br>peak_to_rms=3.592<br>lives in a routing-critical block |
| `block_047.slot_10.block_norm` | 0.592 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0057<br>peak_to_rms=2.291<br>lives in a routing-critical block |
| `block_055.slot_07.block_norm` | 0.592 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0057<br>peak_to_rms=4.568<br>lives in a routing-critical block |
| `block_021.slot_07.block_norm` | 0.594 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0059<br>peak_to_rms=2.989<br>lives in a routing-critical block |
| `block_028.slot_07.block_norm` | 0.594 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0059<br>peak_to_rms=2.597<br>lives in a routing-critical block |
| `block_033.slot_07.block_norm` | 0.594 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0059<br>peak_to_rms=4.069<br>lives in a routing-critical block |
| `block_051.slot_07.block_norm` | 0.594 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0059<br>peak_to_rms=3.999<br>lives in a routing-critical block |
| `block_040.slot_08.block_norm` | 0.594 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0046<br>peak_to_rms=5.904<br>lives in a routing-critical block |
| `block_015.slot_10.block_norm` | 0.594 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0016<br>peak_to_rms=5.950<br>lives in a routing-critical block |
| `block_037.slot_09.block_norm` | 0.595 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0018<br>outlier_fraction=0.0060<br>peak_to_rms=1.534<br>lives in a routing-critical block |
| `block_039.slot_09.block_norm` | 0.595 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0015<br>outlier_fraction=0.0060<br>peak_to_rms=1.509<br>lives in a routing-critical block |
| `block_026.slot_07.block_norm` | 0.596 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0062<br>peak_to_rms=3.163<br>lives in a routing-critical block |
| `block_036.slot_09.block_norm` | 0.597 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0016<br>outlier_fraction=0.0063<br>peak_to_rms=1.454<br>lives in a routing-critical block |
| `block_036.slot_07.block_norm` | 0.597 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0063<br>peak_to_rms=3.043<br>lives in a routing-critical block |
| `block_041.slot_07.block_norm` | 0.597 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0063<br>peak_to_rms=3.606<br>lives in a routing-critical block |
| `block_042.slot_07.block_norm` | 0.597 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0063<br>peak_to_rms=3.518<br>lives in a routing-critical block |
| `block_044.slot_07.block_norm` | 0.597 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0063<br>peak_to_rms=3.731<br>lives in a routing-critical block |
| `block_035.slot_09.block_norm` | 0.598 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0015<br>outlier_fraction=0.0065<br>peak_to_rms=1.450<br>lives in a routing-critical block |
| `block_033.slot_09.block_norm` | 0.598 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0007<br>outlier_fraction=0.0065<br>peak_to_rms=1.461<br>lives in a routing-critical block |
| `block_049.slot_07.block_norm` | 0.598 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0065<br>peak_to_rms=4.444<br>lives in a routing-critical block |
| `block_051.slot_10.block_norm` | 0.598 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0065<br>peak_to_rms=1.433<br>lives in a routing-critical block |
| `block_058.slot_07.block_norm` | 0.598 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0065<br>peak_to_rms=4.487<br>lives in a routing-critical block |
| `block_004.slot_08.block_norm` | 0.598 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0036<br>peak_to_rms=6.534<br>lives in a routing-critical block |
| `block_029.slot_09.block_norm` | 0.599 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0013<br>outlier_fraction=0.0067<br>peak_to_rms=1.480<br>lives in a routing-critical block |
| `block_030.slot_09.block_norm` | 0.599 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0013<br>outlier_fraction=0.0067<br>peak_to_rms=1.478<br>lives in a routing-critical block |
| `block_031.slot_09.block_norm` | 0.599 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0013<br>outlier_fraction=0.0067<br>peak_to_rms=1.462<br>lives in a routing-critical block |
| `block_017.slot_09.block_norm` | 0.599 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0008<br>outlier_fraction=0.0067<br>peak_to_rms=1.531<br>lives in a routing-critical block |
| `block_040.slot_07.block_norm` | 0.599 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0067<br>peak_to_rms=4.931<br>lives in a routing-critical block |
| `block_054.slot_07.block_norm` | 0.599 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0067<br>peak_to_rms=4.307<br>lives in a routing-critical block |
| `block_000.slot_08.block_norm` | 0.600 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0068<br>peak_to_rms=5.462<br>lives in a routing-critical block |
| `block_057.slot_07.block_norm` | 0.600 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0068<br>peak_to_rms=4.236<br>lives in a routing-critical block |
| `block_027.slot_09.block_norm` | 0.601 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0018<br>outlier_fraction=0.0070<br>peak_to_rms=1.512<br>lives in a routing-critical block |
| `block_016.slot_09.block_norm` | 0.601 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0015<br>outlier_fraction=0.0070<br>peak_to_rms=1.484<br>lives in a routing-critical block |
| `block_021.slot_09.block_norm` | 0.603 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0018<br>outlier_fraction=0.0072<br>peak_to_rms=1.571<br>lives in a routing-critical block |
| `block_022.slot_09.block_norm` | 0.603 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0011<br>outlier_fraction=0.0072<br>peak_to_rms=1.550<br>lives in a routing-critical block |
| `block_019.slot_09.block_norm` | 0.603 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0008<br>outlier_fraction=0.0072<br>peak_to_rms=1.580<br>lives in a routing-critical block |
| `block_057.slot_10.block_norm` | 0.603 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0072<br>peak_to_rms=1.291<br>lives in a routing-critical block |
| `block_001.slot_10.block_norm` | 0.603 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0010<br>peak_to_rms=7.184<br>lives in a routing-critical block |
| `block_018.slot_09.block_norm` | 0.604 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0018<br>outlier_fraction=0.0073<br>peak_to_rms=1.477<br>lives in a routing-critical block |
| `block_026.slot_09.block_norm` | 0.604 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0018<br>outlier_fraction=0.0073<br>peak_to_rms=1.502<br>lives in a routing-critical block |
| `block_032.slot_09.block_norm` | 0.604 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0015<br>outlier_fraction=0.0073<br>peak_to_rms=1.448<br>lives in a routing-critical block |
| `block_003.slot_08.block_norm` | 0.604 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0062<br>peak_to_rms=7.319<br>lives in a routing-critical block |
| `block_035.slot_07.block_norm` | 0.604 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0073<br>peak_to_rms=4.525<br>lives in a routing-critical block |
| `block_012.slot_10.block_norm` | 0.604 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0010<br>peak_to_rms=7.378<br>lives in a routing-critical block |
| `block_028.slot_09.block_norm` | 0.605 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0008<br>outlier_fraction=0.0075<br>peak_to_rms=1.473<br>lives in a routing-critical block |
| `block_006.slot_10.block_norm` | 0.606 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0008<br>peak_to_rms=7.624<br>lives in a routing-critical block |
| `block_011.slot_10.block_norm` | 0.606 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0011<br>peak_to_rms=7.666<br>lives in a routing-critical block |
| `block_024.slot_09.block_norm` | 0.607 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0015<br>outlier_fraction=0.0078<br>peak_to_rms=1.540<br>lives in a routing-critical block |
| `block_020.slot_09.block_norm` | 0.607 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0013<br>outlier_fraction=0.0078<br>peak_to_rms=1.541<br>lives in a routing-critical block |
| `block_047.slot_07.block_norm` | 0.607 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0078<br>peak_to_rms=3.522<br>lives in a routing-critical block |
| `block_000.slot_07.block_norm` | 0.607 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0054<br>peak_to_rms=7.827<br>lives in a routing-critical block |
| `block_055.slot_10.block_norm` | 0.608 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0080<br>peak_to_rms=1.463<br>lives in a routing-critical block |
| `block_025.slot_09.block_norm` | 0.609 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0016<br>outlier_fraction=0.0081<br>peak_to_rms=1.494<br>lives in a routing-critical block |
| `block_032.slot_07.block_norm` | 0.609 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0081<br>peak_to_rms=2.783<br>lives in a routing-critical block |
| `block_008.slot_10.block_norm` | 0.610 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0003<br>peak_to_rms=8.269<br>lives in a routing-critical block |
| `block_001.slot_08.block_norm` | 0.611 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0002<br>outlier_fraction=0.0005<br>peak_to_rms=8.334<br>lives in a routing-critical block |
| `block_000.slot_09.block_norm` | 0.611 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0002<br>outlier_fraction=0.0063<br>peak_to_rms=8.419<br>lives in a routing-critical block |
| `block_004.slot_10.block_norm` | 0.613 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0011<br>peak_to_rms=8.618<br>lives in a routing-critical block |
| `block_052.slot_10.block_norm` | 0.613 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0086<br>peak_to_rms=1.423<br>lives in a routing-critical block |
| `block_054.slot_10.block_norm` | 0.613 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0086<br>peak_to_rms=1.622<br>lives in a routing-critical block |
| `block_010.slot_10.block_norm` | 0.617 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0003<br>peak_to_rms=9.166<br>lives in a routing-critical block |
| `block_012.slot_08.block_norm` | 0.618 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0013<br>peak_to_rms=9.381<br>lives in a routing-critical block |
| `block_053.slot_10.block_norm` | 0.619 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0094<br>peak_to_rms=1.452<br>lives in a routing-critical block |
| `block_002.slot_08.block_norm` | 0.620 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0055<br>peak_to_rms=9.666<br>lives in a routing-critical block |
| `block_023.slot_10.block_norm` | 0.621 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0007<br>peak_to_rms=9.753<br>lives in a routing-critical block |
| `block_023.slot_08.block_norm` | 0.622 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0002<br>outlier_fraction=0.0018<br>peak_to_rms=9.928<br>lives in a routing-critical block |
| `block_056.slot_08.block_norm` | 0.628 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0044<br>peak_to_rms=10.721<br>lives in a routing-critical block |
| `block_034.slot_10.block_norm` | 0.628 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0010<br>peak_to_rms=10.736<br>lives in a routing-critical block |
| `block_034.slot_08.block_norm` | 0.631 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0003<br>outlier_fraction=0.0037<br>peak_to_rms=11.269<br>lives in a routing-critical block |
| `block_003.slot_10.block_norm` | 0.634 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0010<br>peak_to_rms=11.593<br>lives in a routing-critical block |
| `block_000.slot_10.block_norm` | 0.636 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0080<br>peak_to_rms=11.967<br>lives in a routing-critical block |
| `block_056.slot_10.block_norm` | 0.643 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0007<br>peak_to_rms=12.923<br>lives in a routing-critical block |
| `block_045.slot_08.block_norm` | 0.644 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0034<br>peak_to_rms=13.031<br>lives in a routing-critical block |
| `block_002.slot_10.block_norm` | 0.644 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0007<br>peak_to_rms=13.036<br>lives in a routing-critical block |
| `block_045.slot_10.block_norm` | 0.646 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0003<br>peak_to_rms=13.365<br>lives in a routing-critical block |
| `block_062.slot_10.block_norm` | 0.646 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0007<br>peak_to_rms=13.427<br>lives in a routing-critical block |
| `block_061.slot_08.block_norm` | 0.648 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0033<br>peak_to_rms=13.596<br>lives in a routing-critical block |
| `block_063.slot_10.block_norm` | 0.648 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0007<br>peak_to_rms=13.648<br>lives in a routing-critical block |
| `block_063.slot_08.block_norm` | 0.648 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0024<br>peak_to_rms=13.648<br>lives in a routing-critical block |
| `block_061.slot_10.block_norm` | 0.658 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0008<br>peak_to_rms=15.031<br>lives in a routing-critical block |
| `block_062.slot_08.block_norm` | 0.659 | distribution=dense_balanced<br>sampled_values=6144/6144<br>zero_fraction=0.0000<br>near_zero_fraction=0.0000<br>outlier_fraction=0.0016<br>peak_to_rms=15.218<br>lives in a routing-critical block |

## Deferred tensors

| Tensor | Kind | Disposition | Reasons |
| ------ | ---- | ----------- | ------- |
| `embedding.slot_00.token_embedding` | token_embedding | observe_only | distribution=dense_balanced<br>sampled_values=65536/805306368<br>zero_fraction=0.0000<br>near_zero_fraction=0.0675<br>outlier_fraction=0.0000<br>peak_to_rms=4.722 |

## Highest-risk tensors

| Tensor | Region | Risk | Reasons |
| ------ | ------ | ---: | ------- |
| `block_063.slot_11.router` | routing_critical | 0.727 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0470<br>outlier_fraction=0.0009<br>peak_to_rms=15.632<br>linked to routing structure |
| `block_011.slot_11.router` | routing_critical | 0.727 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0238<br>outlier_fraction=0.0005<br>peak_to_rms=15.616<br>linked to routing structure |
| `block_033.slot_11.router` | routing_critical | 0.726 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0300<br>outlier_fraction=0.0005<br>peak_to_rms=15.542<br>linked to routing structure |
| `block_062.slot_11.router` | routing_critical | 0.717 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0208<br>outlier_fraction=0.0011<br>peak_to_rms=14.233<br>linked to routing structure |
| `block_047.slot_11.router` | routing_critical | 0.715 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0462<br>outlier_fraction=0.0003<br>peak_to_rms=13.935<br>linked to routing structure |
| `block_031.slot_11.router` | routing_critical | 0.712 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0335<br>outlier_fraction=0.0006<br>peak_to_rms=13.529<br>linked to routing structure |
| `block_007.slot_11.router` | routing_critical | 0.710 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.1395<br>outlier_fraction=0.0005<br>peak_to_rms=13.223<br>linked to routing structure |
| `block_005.slot_11.router` | routing_critical | 0.709 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0224<br>outlier_fraction=0.0007<br>peak_to_rms=13.139<br>linked to routing structure |
| `block_003.slot_11.router` | routing_critical | 0.706 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0388<br>outlier_fraction=0.0013<br>peak_to_rms=12.607<br>linked to routing structure |
| `block_044.slot_11.router` | routing_critical | 0.704 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0528<br>outlier_fraction=0.0002<br>peak_to_rms=12.375<br>linked to routing structure |
| `block_061.slot_11.router` | routing_critical | 0.704 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0243<br>outlier_fraction=0.0013<br>peak_to_rms=12.314<br>linked to routing structure |
| `block_034.slot_11.router` | routing_critical | 0.704 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0466<br>outlier_fraction=0.0015<br>peak_to_rms=12.307<br>linked to routing structure |
| `block_002.slot_11.router` | routing_critical | 0.701 | distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0247<br>outlier_fraction=0.0005<br>peak_to_rms=11.872<br>linked to routing structure |

## Layer readiness

| Label | Block | Routing critical | Candidate targets | Mean readiness | Max risk |
| ----- | ----: | ---------------- | ----------------: | -------------: | -------: |
| unassigned | - | no | 0 | 0.107 | 0.570 |
| block_000 | 0 | yes | 7 | 0.131 | 0.662 |
| block_001 | 1 | yes | 7 | 0.132 | 0.671 |
| block_002 | 2 | yes | 7 | 0.131 | 0.701 |
| block_003 | 3 | yes | 7 | 0.131 | 0.706 |
| block_004 | 4 | yes | 7 | 0.132 | 0.687 |
| block_005 | 5 | yes | 7 | 0.132 | 0.709 |
| block_006 | 6 | yes | 7 | 0.133 | 0.671 |
| block_007 | 7 | yes | 7 | 0.133 | 0.710 |
| block_008 | 8 | yes | 7 | 0.133 | 0.682 |
| block_009 | 9 | yes | 7 | 0.132 | 0.690 |
| block_010 | 10 | yes | 7 | 0.132 | 0.681 |
| block_011 | 11 | yes | 7 | 0.132 | 0.727 |
| block_012 | 12 | yes | 7 | 0.132 | 0.695 |
| block_013 | 13 | yes | 7 | 0.133 | 0.676 |
| block_014 | 14 | yes | 7 | 0.133 | 0.683 |
| block_015 | 15 | yes | 7 | 0.132 | 0.687 |
| block_016 | 16 | yes | 7 | 0.132 | 0.683 |
| block_017 | 17 | yes | 7 | 0.132 | 0.684 |
| block_018 | 18 | yes | 7 | 0.132 | 0.680 |
| block_019 | 19 | yes | 7 | 0.132 | 0.691 |
| block_020 | 20 | yes | 7 | 0.132 | 0.694 |
| block_021 | 21 | yes | 7 | 0.132 | 0.680 |
| block_022 | 22 | yes | 7 | 0.132 | 0.679 |
| block_023 | 23 | yes | 7 | 0.131 | 0.699 |
| block_024 | 24 | yes | 7 | 0.132 | 0.684 |
| block_025 | 25 | yes | 7 | 0.132 | 0.686 |
| block_026 | 26 | yes | 7 | 0.132 | 0.682 |
| block_027 | 27 | yes | 7 | 0.132 | 0.690 |
| block_028 | 28 | yes | 7 | 0.132 | 0.686 |
| block_029 | 29 | yes | 7 | 0.132 | 0.699 |
| block_030 | 30 | yes | 7 | 0.132 | 0.674 |
| block_031 | 31 | yes | 7 | 0.132 | 0.712 |
| block_032 | 32 | yes | 7 | 0.132 | 0.699 |
| block_033 | 33 | yes | 7 | 0.132 | 0.726 |
| block_034 | 34 | yes | 7 | 0.131 | 0.704 |
| block_035 | 35 | yes | 7 | 0.132 | 0.679 |
| block_036 | 36 | yes | 7 | 0.132 | 0.687 |
| block_037 | 37 | yes | 7 | 0.132 | 0.685 |
| block_038 | 38 | yes | 7 | 0.132 | 0.688 |
| block_039 | 39 | yes | 7 | 0.132 | 0.685 |
| block_040 | 40 | yes | 7 | 0.132 | 0.688 |
| block_041 | 41 | yes | 7 | 0.132 | 0.689 |
| block_042 | 42 | yes | 7 | 0.132 | 0.684 |
| block_043 | 43 | yes | 7 | 0.132 | 0.672 |
| block_044 | 44 | yes | 7 | 0.132 | 0.704 |
| block_045 | 45 | yes | 7 | 0.131 | 0.692 |
| block_046 | 46 | yes | 7 | 0.132 | 0.686 |
| block_047 | 47 | yes | 7 | 0.132 | 0.715 |
| block_048 | 48 | yes | 7 | 0.132 | 0.677 |
| block_049 | 49 | yes | 7 | 0.132 | 0.672 |
| block_050 | 50 | yes | 7 | 0.132 | 0.691 |
| block_051 | 51 | yes | 7 | 0.132 | 0.685 |
| block_052 | 52 | yes | 7 | 0.132 | 0.672 |
| block_053 | 53 | yes | 7 | 0.132 | 0.678 |
| block_054 | 54 | yes | 7 | 0.132 | 0.668 |
| block_055 | 55 | yes | 7 | 0.132 | 0.685 |
| block_056 | 56 | yes | 7 | 0.131 | 0.697 |
| block_057 | 57 | yes | 7 | 0.132 | 0.676 |
| block_058 | 58 | yes | 7 | 0.132 | 0.669 |
| block_059 | 59 | yes | 7 | 0.132 | 0.673 |
| block_060 | 60 | yes | 7 | 0.133 | 0.651 |
| block_061 | 61 | yes | 7 | 0.131 | 0.704 |
| block_062 | 62 | yes | 7 | 0.131 | 0.717 |
| block_063 | 63 | yes | 7 | 0.131 | 0.727 |

## Notes

- Top quantization candidate is `block_030.slot_01.moe_expert.down` with readiness 0.188.
- 64 tensors are classified as routing-critical and should be handled cautiously.
- Routing analysis notes: Primary routing candidates are plain f32 tensors oriented from d_model to expert logits. Primary routing candidates occupy a stable block slot (11) across observed blocks. Observed primary routing tensors match the Grok-style router shape `(6144, 8)`.
