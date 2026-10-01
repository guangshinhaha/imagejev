# imagejev:pilot_v1_best

Evaluated on a fixed subset of at most 300 images per split.

Calibration fitted on `val-tasks`: choice=0.88, score=20.00, bool=3.01, bool_bias=0.11

- cold latency: p50 73.8 ms, p95 91.0 ms (n=30)
- warm latency: p50 11.0 ms, p95 18.9 ms (n=30)

### val-tasks (406 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 106 | 0.547 | - | 0.699 | 0.140 | 0.565 |
| document | choice | 119 | 0.580 | - | 1.025 | 0.128 | 0.615 |
| document | score | 18 | 0.333 | 0.611 | 1.378 | 0.077 | 0.333 |
| photo | bool | 48 | 0.771 | - | 0.510 | 0.121 | 0.821 |
| photo | score | 16 | 0.250 | 0.500 | 1.398 | 0.009 | 0.231 |
| screenshot | bool | 79 | 0.797 | - | 0.524 | 0.224 | 0.891 |
| screenshot | score | 20 | 0.350 | 0.850 | 1.385 | 0.091 | 0.375 |
| document | all | 243 | 0.547 | - | 0.909 | 0.116 | 0.574 |
| photo | all | 64 | 0.641 | - | 0.732 | 0.093 | 0.750 |
| screenshot | all | 99 | 0.707 | - | 0.698 | 0.197 | 0.787 |
| all | bool | 233 | 0.678 | - | 0.601 | 0.032 | 0.717 |
| all | choice | 119 | 0.580 | - | 1.025 | 0.128 | 0.615 |
| all | score | 54 | 0.315 | 0.667 | 1.386 | 0.056 | 0.341 |
| all | all | 406 | 0.601 | - | 0.830 | 0.056 | 0.662 |

### test-images (1079 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 284 | 0.609 | - | 0.671 | 0.084 | 0.632 |
| document | choice | 93 | 0.935 | - | 0.256 | 0.117 | 0.960 |
| document | score | 67 | 0.313 | 0.821 | 1.358 | 0.071 | 0.333 |
| photo | bool | 101 | 0.713 | - | 0.589 | 0.102 | 0.778 |
| photo | choice | 46 | 0.870 | - | 0.414 | 0.101 | 0.946 |
| photo | score | 72 | 0.500 | 0.931 | 1.344 | 0.232 | 0.500 |
| screenshot | bool | 301 | 0.565 | - | 0.667 | 0.065 | 0.614 |
| screenshot | choice | 98 | 0.939 | - | 0.295 | 0.135 | 0.975 |
| screenshot | score | 17 | 0.706 | 1.000 | 1.260 | 0.482 | 0.786 |
| document | all | 444 | 0.633 | - | 0.688 | 0.066 | 0.702 |
| photo | all | 219 | 0.676 | - | 0.800 | 0.136 | 0.716 |
| screenshot | all | 416 | 0.659 | - | 0.604 | 0.073 | 0.697 |
| all | bool | 686 | 0.605 | - | 0.657 | 0.057 | 0.645 |
| all | choice | 237 | 0.924 | - | 0.303 | 0.102 | 0.968 |
| all | score | 156 | 0.442 | 0.891 | 1.341 | 0.175 | 0.488 |
| all | all | 1079 | 0.652 | - | 0.678 | 0.060 | 0.698 |

### test-tasks (401 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 24 | 0.542 | - | 0.816 | 0.213 | 0.450 |
| document | choice | 32 | 1.000 | - | 0.260 | 0.196 | 1.000 |
| document | score | 70 | 0.243 | 0.529 | 1.385 | 0.013 | 0.268 |
| photo | bool | 8 | 0.750 | - | 0.430 | 0.219 | 0.857 |
| photo | choice | 58 | 0.655 | - | 2.549 | 0.317 | 0.638 |
| photo | score | 24 | 0.250 | 0.500 | 1.387 | 0.006 | 0.200 |
| screenshot | bool | 38 | 0.684 | - | 0.603 | 0.210 | 0.645 |
| screenshot | score | 147 | 0.054 | 0.776 | 1.394 | 0.204 | 0.025 |
| document | all | 126 | 0.492 | - | 0.991 | 0.075 | 0.574 |
| photo | all | 90 | 0.556 | - | 2.051 | 0.196 | 0.625 |
| screenshot | all | 185 | 0.184 | - | 1.231 | 0.205 | 0.189 |
| all | bool | 70 | 0.643 | - | 0.656 | 0.144 | 0.625 |
| all | choice | 90 | 0.778 | - | 1.735 | 0.229 | 0.764 |
| all | score | 241 | 0.129 | 0.676 | 1.391 | 0.129 | 0.109 |
| all | all | 401 | 0.364 | - | 1.340 | 0.141 | 0.414 |

### test-styles (1202 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 451 | 0.588 | - | 0.708 | 0.099 | 0.601 |
| document | choice | 133 | 0.872 | - | 0.336 | 0.089 | 0.935 |
| document | score | 169 | 0.266 | 0.769 | 1.368 | 0.023 | 0.272 |
| screenshot | bool | 318 | 0.604 | - | 0.666 | 0.018 | 0.624 |
| screenshot | choice | 106 | 0.915 | - | 0.347 | 0.142 | 0.988 |
| screenshot | score | 25 | 0.680 | 1.000 | 1.249 | 0.391 | 0.750 |
| document | all | 753 | 0.566 | - | 0.790 | 0.062 | 0.643 |
| screenshot | all | 449 | 0.682 | - | 0.623 | 0.053 | 0.706 |
| all | bool | 769 | 0.594 | - | 0.691 | 0.062 | 0.609 |
| all | choice | 239 | 0.891 | - | 0.341 | 0.109 | 0.964 |
| all | score | 194 | 0.320 | 0.799 | 1.352 | 0.057 | 0.333 |
| all | all | 1202 | 0.609 | - | 0.728 | 0.044 | 0.668 |

### test-external (689 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 41 | 0.463 | - | 1.089 | 0.380 | 0.455 |
| document | score | 33 | 0.485 | 0.818 | 1.367 | 0.219 | 0.519 |
| photo | bool | 174 | 0.868 | - | 0.324 | 0.082 | 0.936 |
| photo | choice | 348 | 0.974 | - | 0.216 | 0.147 | 0.996 |
| screenshot | choice | 93 | 0.892 | - | 0.410 | 0.090 | 0.947 |
| document | all | 74 | 0.473 | - | 1.213 | 0.309 | 0.483 |
| photo | all | 522 | 0.939 | - | 0.252 | 0.118 | 0.976 |
| screenshot | all | 93 | 0.892 | - | 0.410 | 0.090 | 0.947 |
| all | bool | 215 | 0.791 | - | 0.470 | 0.066 | 0.837 |
| all | choice | 441 | 0.957 | - | 0.257 | 0.131 | 0.986 |
| all | score | 33 | 0.485 | 0.818 | 1.367 | 0.219 | 0.519 |
| all | all | 689 | 0.882 | - | 0.377 | 0.091 | 0.933 |
