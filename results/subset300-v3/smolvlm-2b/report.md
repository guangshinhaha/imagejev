# smolvlm:SmolVLM-Instruct

Evaluated on a fixed subset of at most 300 images per split.

Calibration fitted on `val-tasks`: choice=0.96, score=11.47, bool=6.86, bool_bias=0.02

- cold latency: p50 346.7 ms, p95 351.7 ms (n=30)
- warm latency: p50 112.1 ms, p95 115.1 ms (n=30)

### val-tasks (406 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 106 | 0.491 | - | 0.706 | 0.048 | 0.459 |
| document | choice | 119 | 0.882 | - | 0.365 | 0.057 | 0.938 |
| document | score | 18 | 0.389 | 0.556 | 1.374 | 0.101 | 0.333 |
| photo | bool | 48 | 0.646 | - | 0.598 | 0.151 | 0.692 |
| photo | score | 16 | 0.188 | 0.375 | 1.373 | 0.107 | 0.231 |
| screenshot | bool | 79 | 0.633 | - | 0.685 | 0.155 | 0.594 |
| screenshot | score | 20 | 0.250 | 0.600 | 1.394 | 0.073 | 0.312 |
| document | all | 243 | 0.675 | - | 0.588 | 0.046 | 0.723 |
| photo | all | 64 | 0.531 | - | 0.792 | 0.140 | 0.635 |
| screenshot | all | 99 | 0.556 | - | 0.828 | 0.139 | 0.637 |
| all | bool | 233 | 0.571 | - | 0.677 | 0.047 | 0.561 |
| all | choice | 119 | 0.882 | - | 0.365 | 0.057 | 0.938 |
| all | score | 54 | 0.278 | 0.519 | 1.381 | 0.031 | 0.318 |
| all | all | 406 | 0.623 | - | 0.679 | 0.037 | 0.674 |

### test-images (1489 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 277 | 0.679 | - | 0.621 | 0.115 | 0.734 |
| document | choice | 93 | 0.720 | - | 0.649 | 0.141 | 0.787 |
| document | score | 212 | 0.255 | 0.613 | 1.412 | 0.040 | 0.265 |
| photo | bool | 104 | 0.865 | - | 0.487 | 0.231 | 0.917 |
| photo | choice | 46 | 0.630 | - | 1.018 | 0.236 | 0.649 |
| photo | score | 150 | 0.287 | 0.627 | 1.365 | 0.128 | 0.317 |
| screenshot | bool | 296 | 0.588 | - | 0.654 | 0.053 | 0.620 |
| screenshot | choice | 208 | 0.505 | - | 1.159 | 0.248 | 0.515 |
| screenshot | score | 103 | 0.204 | 0.485 | 1.456 | 0.124 | 0.181 |
| document | all | 582 | 0.531 | - | 0.913 | 0.074 | 0.616 |
| photo | all | 300 | 0.540 | - | 1.007 | 0.159 | 0.633 |
| screenshot | all | 607 | 0.494 | - | 0.963 | 0.101 | 0.558 |
| all | bool | 677 | 0.668 | - | 0.615 | 0.103 | 0.723 |
| all | choice | 347 | 0.579 | - | 1.004 | 0.206 | 0.604 |
| all | score | 465 | 0.254 | 0.589 | 1.406 | 0.056 | 0.269 |
| all | all | 1489 | 0.518 | - | 0.953 | 0.080 | 0.592 |

### test-tasks (395 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 22 | 0.409 | - | 0.706 | 0.116 | 0.500 |
| document | choice | 32 | 0.656 | - | 1.132 | 0.250 | 0.692 |
| document | score | 70 | 0.157 | 0.429 | 1.421 | 0.155 | 0.161 |
| photo | bool | 7 | 0.714 | - | 0.662 | 0.194 | 0.667 |
| photo | choice | 58 | 0.793 | - | 0.627 | 0.107 | 0.787 |
| photo | score | 24 | 0.333 | 0.583 | 1.379 | 0.052 | 0.350 |
| screenshot | bool | 35 | 0.714 | - | 0.625 | 0.204 | 0.821 |
| screenshot | score | 147 | 0.429 | 0.605 | 1.341 | 0.128 | 0.492 |
| document | all | 124 | 0.331 | - | 1.220 | 0.157 | 0.390 |
| photo | all | 89 | 0.663 | - | 0.833 | 0.099 | 0.750 |
| screenshot | all | 182 | 0.484 | - | 1.204 | 0.142 | 0.562 |
| all | bool | 64 | 0.609 | - | 0.657 | 0.135 | 0.692 |
| all | choice | 90 | 0.744 | - | 0.807 | 0.130 | 0.764 |
| all | score | 241 | 0.340 | 0.552 | 1.368 | 0.036 | 0.378 |
| all | all | 395 | 0.476 | - | 1.125 | 0.051 | 0.528 |
