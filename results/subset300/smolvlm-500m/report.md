# smolvlm:SmolVLM-500M-Instruct

Evaluated on a fixed subset of at most 300 images per split.

Calibration fitted on `val-tasks`: choice=1.78, score=20.00, bool=4.61, bool_bias=0.32

- cold latency: p50 241.4 ms, p95 272.6 ms (n=30)
- warm latency: p50 109.7 ms, p95 130.3 ms (n=30)

### val-tasks (406 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 106 | 0.566 | - | 0.669 | 0.053 | 0.600 |
| document | choice | 119 | 0.479 | - | 1.151 | 0.095 | 0.531 |
| document | score | 18 | 0.333 | 0.444 | 1.364 | 0.061 | 0.333 |
| photo | bool | 48 | 0.646 | - | 0.616 | 0.144 | 0.641 |
| photo | score | 16 | 0.188 | 0.625 | 1.387 | 0.079 | 0.231 |
| screenshot | bool | 79 | 0.608 | - | 0.639 | 0.126 | 0.625 |
| screenshot | score | 20 | 0.200 | 0.600 | 1.405 | 0.076 | 0.188 |
| document | all | 243 | 0.506 | - | 0.956 | 0.069 | 0.549 |
| photo | all | 64 | 0.531 | - | 0.809 | 0.128 | 0.596 |
| screenshot | all | 99 | 0.525 | - | 0.793 | 0.116 | 0.613 |
| all | bool | 233 | 0.597 | - | 0.648 | 0.066 | 0.610 |
| all | choice | 119 | 0.479 | - | 1.151 | 0.095 | 0.531 |
| all | score | 54 | 0.241 | 0.556 | 1.386 | 0.031 | 0.227 |
| all | all | 406 | 0.515 | - | 0.893 | 0.058 | 0.575 |

### test-images (1079 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 284 | 0.560 | - | 0.692 | 0.058 | 0.566 |
| document | choice | 93 | 0.806 | - | 0.672 | 0.220 | 0.880 |
| document | score | 67 | 0.254 | 0.612 | 1.393 | 0.020 | 0.259 |
| photo | bool | 101 | 0.871 | - | 0.432 | 0.165 | 0.914 |
| photo | choice | 46 | 0.609 | - | 1.013 | 0.211 | 0.595 |
| photo | score | 72 | 0.167 | 0.514 | 1.407 | 0.133 | 0.138 |
| screenshot | bool | 301 | 0.555 | - | 0.711 | 0.103 | 0.539 |
| screenshot | choice | 98 | 0.296 | - | 1.938 | 0.324 | 0.304 |
| screenshot | score | 17 | 0.353 | 0.588 | 1.356 | 0.054 | 0.429 |
| document | all | 444 | 0.565 | - | 0.793 | 0.049 | 0.626 |
| photo | all | 219 | 0.584 | - | 0.874 | 0.126 | 0.676 |
| screenshot | all | 416 | 0.486 | - | 1.026 | 0.120 | 0.498 |
| all | bool | 686 | 0.603 | - | 0.662 | 0.040 | 0.617 |
| all | choice | 237 | 0.557 | - | 1.262 | 0.078 | 0.574 |
| all | score | 156 | 0.224 | 0.564 | 1.395 | 0.068 | 0.216 |
| all | all | 1079 | 0.538 | - | 0.900 | 0.048 | 0.600 |

### test-tasks (401 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 24 | 0.417 | - | 0.762 | 0.279 | 0.300 |
| document | choice | 32 | 0.406 | - | 1.607 | 0.148 | 0.423 |
| document | score | 70 | 0.314 | 0.600 | 1.386 | 0.041 | 0.321 |
| photo | bool | 8 | 0.500 | - | 0.726 | 0.202 | 0.429 |
| photo | choice | 58 | 0.379 | - | 1.238 | 0.211 | 0.426 |
| photo | score | 24 | 0.208 | 0.417 | 1.396 | 0.078 | 0.250 |
| screenshot | bool | 38 | 0.763 | - | 0.558 | 0.158 | 0.839 |
| screenshot | score | 147 | 0.184 | 0.456 | 1.409 | 0.102 | 0.203 |
| document | all | 126 | 0.357 | - | 1.323 | 0.094 | 0.356 |
| photo | all | 90 | 0.344 | - | 1.235 | 0.151 | 0.403 |
| screenshot | all | 185 | 0.303 | - | 1.234 | 0.114 | 0.351 |
| all | bool | 70 | 0.614 | - | 0.647 | 0.129 | 0.607 |
| all | choice | 90 | 0.389 | - | 1.369 | 0.126 | 0.444 |
| all | score | 241 | 0.224 | 0.494 | 1.401 | 0.056 | 0.223 |
| all | all | 401 | 0.329 | - | 1.262 | 0.062 | 0.361 |

### test-styles (1202 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 451 | 0.557 | - | 0.673 | 0.062 | 0.584 |
| document | choice | 133 | 0.707 | - | 0.841 | 0.176 | 0.757 |
| document | score | 169 | 0.337 | 0.645 | 1.382 | 0.066 | 0.360 |
| screenshot | bool | 318 | 0.560 | - | 0.679 | 0.045 | 0.584 |
| screenshot | choice | 106 | 0.340 | - | 1.803 | 0.320 | 0.400 |
| screenshot | score | 25 | 0.280 | 0.400 | 1.354 | 0.011 | 0.350 |
| document | all | 753 | 0.534 | - | 0.862 | 0.052 | 0.585 |
| screenshot | all | 449 | 0.492 | - | 0.982 | 0.098 | 0.528 |
| all | bool | 769 | 0.558 | - | 0.675 | 0.044 | 0.586 |
| all | choice | 239 | 0.544 | - | 1.268 | 0.141 | 0.589 |
| all | score | 194 | 0.330 | 0.613 | 1.378 | 0.056 | 0.346 |
| all | all | 1202 | 0.518 | - | 0.907 | 0.042 | 0.563 |

### test-external (689 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 41 | 0.439 | - | 0.871 | 0.303 | 0.424 |
| document | score | 33 | 0.242 | 0.333 | 1.364 | 0.030 | 0.259 |
| photo | bool | 174 | 0.695 | - | 0.599 | 0.077 | 0.714 |
| photo | choice | 348 | 0.448 | - | 1.179 | 0.274 | 0.487 |
| screenshot | choice | 93 | 0.333 | - | 1.534 | 0.197 | 0.333 |
| document | all | 74 | 0.351 | - | 1.091 | 0.181 | 0.400 |
| photo | all | 522 | 0.531 | - | 0.985 | 0.161 | 0.574 |
| screenshot | all | 93 | 0.333 | - | 1.534 | 0.197 | 0.333 |
| all | bool | 215 | 0.647 | - | 0.651 | 0.090 | 0.645 |
| all | choice | 441 | 0.424 | - | 1.254 | 0.255 | 0.462 |
| all | score | 33 | 0.242 | 0.333 | 1.364 | 0.030 | 0.259 |
| all | all | 689 | 0.485 | - | 1.071 | 0.165 | 0.538 |
