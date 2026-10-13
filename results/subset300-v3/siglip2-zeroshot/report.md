# siglip2-zeroshot

Evaluated on a fixed subset of at most 300 images per split.

Calibration fitted on `val-tasks`: choice=0.73, score=2.38, bool=3.08, bool_bias=-0.89

- cold latency: p50 36.5 ms, p95 38.0 ms (n=30)
- warm latency: p50 0.8 ms, p95 1.0 ms (n=30)

### val-tasks (406 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 106 | 0.519 | - | 0.703 | 0.171 | 0.553 |
| document | choice | 119 | 0.588 | - | 0.987 | 0.120 | 0.615 |
| document | score | 18 | 0.389 | 0.722 | 1.335 | 0.163 | 0.400 |
| photo | bool | 48 | 0.750 | - | 0.509 | 0.088 | 0.846 |
| photo | score | 16 | 0.312 | 0.625 | 1.418 | 0.065 | 0.308 |
| screenshot | bool | 79 | 0.823 | - | 0.522 | 0.193 | 0.875 |
| screenshot | score | 20 | 0.350 | 0.850 | 1.356 | 0.084 | 0.438 |
| document | all | 243 | 0.543 | - | 0.889 | 0.121 | 0.569 |
| photo | all | 64 | 0.641 | - | 0.736 | 0.082 | 0.731 |
| screenshot | all | 99 | 0.727 | - | 0.690 | 0.171 | 0.812 |
| all | bool | 233 | 0.670 | - | 0.601 | 0.053 | 0.701 |
| all | choice | 119 | 0.588 | - | 0.987 | 0.120 | 0.615 |
| all | score | 54 | 0.352 | 0.741 | 1.368 | 0.073 | 0.364 |
| all | all | 406 | 0.603 | - | 0.816 | 0.071 | 0.658 |

### test-images (1489 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 277 | 0.599 | - | 0.691 | 0.058 | 0.613 |
| document | choice | 93 | 0.860 | - | 0.600 | 0.124 | 0.880 |
| document | score | 212 | 0.231 | 0.599 | 1.381 | 0.067 | 0.229 |
| photo | bool | 104 | 0.596 | - | 0.709 | 0.133 | 0.655 |
| photo | choice | 46 | 0.739 | - | 0.916 | 0.134 | 0.865 |
| photo | score | 150 | 0.353 | 0.733 | 1.367 | 0.084 | 0.350 |
| screenshot | bool | 296 | 0.557 | - | 0.693 | 0.055 | 0.565 |
| screenshot | choice | 208 | 0.760 | - | 0.560 | 0.093 | 0.820 |
| screenshot | score | 103 | 0.243 | 0.612 | 1.395 | 0.054 | 0.277 |
| document | all | 582 | 0.507 | - | 0.928 | 0.065 | 0.575 |
| photo | all | 300 | 0.497 | - | 1.070 | 0.100 | 0.529 |
| screenshot | all | 607 | 0.573 | - | 0.766 | 0.058 | 0.652 |
| all | bool | 677 | 0.581 | - | 0.694 | 0.063 | 0.590 |
| all | choice | 347 | 0.784 | - | 0.618 | 0.070 | 0.842 |
| all | score | 465 | 0.273 | 0.645 | 1.380 | 0.023 | 0.280 |
| all | all | 1489 | 0.532 | - | 0.891 | 0.043 | 0.597 |

### test-tasks (395 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 22 | 0.455 | - | 0.882 | 0.302 | 0.389 |
| document | choice | 32 | 0.906 | - | 0.345 | 0.120 | 1.000 |
| document | score | 70 | 0.286 | 0.643 | 1.379 | 0.021 | 0.304 |
| photo | bool | 7 | 0.857 | - | 0.326 | 0.141 | 1.000 |
| photo | choice | 58 | 0.638 | - | 1.260 | 0.207 | 0.681 |
| photo | score | 24 | 0.292 | 0.792 | 1.379 | 0.084 | 0.350 |
| screenshot | bool | 35 | 0.714 | - | 0.558 | 0.116 | 0.714 |
| screenshot | score | 147 | 0.204 | 0.497 | 1.398 | 0.085 | 0.229 |
| document | all | 124 | 0.476 | - | 1.024 | 0.058 | 0.530 |
| photo | all | 89 | 0.562 | - | 1.219 | 0.146 | 0.625 |
| screenshot | all | 182 | 0.302 | - | 1.237 | 0.091 | 0.349 |
| all | bool | 64 | 0.641 | - | 0.644 | 0.109 | 0.654 |
| all | choice | 90 | 0.733 | - | 0.935 | 0.141 | 0.806 |
| all | score | 241 | 0.237 | 0.568 | 1.391 | 0.051 | 0.264 |
| all | all | 395 | 0.415 | - | 1.166 | 0.070 | 0.475 |
