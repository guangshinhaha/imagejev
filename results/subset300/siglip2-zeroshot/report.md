# siglip2-zeroshot

Evaluated on a fixed subset of at most 300 images per split.

Calibration fitted on `val-tasks`: choice=0.73, score=2.40, bool=3.08, bool_bias=-0.89

- cold latency: p50 69.2 ms, p95 76.5 ms (n=30)
- warm latency: p50 0.8 ms, p95 1.1 ms (n=30)

### val-tasks (406 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 106 | 0.519 | - | 0.703 | 0.171 | 0.553 |
| document | choice | 119 | 0.588 | - | 0.987 | 0.120 | 0.615 |
| document | score | 18 | 0.389 | 0.722 | 1.336 | 0.163 | 0.400 |
| photo | bool | 48 | 0.750 | - | 0.509 | 0.088 | 0.846 |
| photo | score | 16 | 0.312 | 0.625 | 1.418 | 0.050 | 0.308 |
| screenshot | bool | 79 | 0.823 | - | 0.521 | 0.193 | 0.875 |
| screenshot | score | 20 | 0.350 | 0.850 | 1.357 | 0.084 | 0.438 |
| document | all | 243 | 0.543 | - | 0.889 | 0.121 | 0.569 |
| photo | all | 64 | 0.641 | - | 0.736 | 0.079 | 0.731 |
| screenshot | all | 99 | 0.727 | - | 0.690 | 0.171 | 0.812 |
| all | bool | 233 | 0.670 | - | 0.601 | 0.056 | 0.701 |
| all | choice | 119 | 0.588 | - | 0.987 | 0.120 | 0.615 |
| all | score | 54 | 0.352 | 0.741 | 1.368 | 0.073 | 0.364 |
| all | all | 406 | 0.603 | - | 0.816 | 0.073 | 0.658 |

### test-images (1079 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 284 | 0.599 | - | 0.690 | 0.059 | 0.614 |
| document | choice | 93 | 0.860 | - | 0.600 | 0.124 | 0.880 |
| document | score | 67 | 0.239 | 0.761 | 1.343 | 0.065 | 0.241 |
| photo | bool | 101 | 0.624 | - | 0.665 | 0.112 | 0.679 |
| photo | choice | 46 | 0.739 | - | 0.916 | 0.134 | 0.865 |
| photo | score | 72 | 0.306 | 0.806 | 1.357 | 0.097 | 0.259 |
| screenshot | bool | 301 | 0.548 | - | 0.699 | 0.066 | 0.552 |
| screenshot | choice | 98 | 0.796 | - | 0.672 | 0.068 | 0.886 |
| screenshot | score | 17 | 0.176 | 0.647 | 1.425 | 0.209 | 0.214 |
| document | all | 444 | 0.599 | - | 0.770 | 0.066 | 0.660 |
| photo | all | 219 | 0.543 | - | 0.945 | 0.099 | 0.591 |
| screenshot | all | 416 | 0.591 | - | 0.723 | 0.062 | 0.634 |
| all | bool | 686 | 0.580 | - | 0.690 | 0.063 | 0.592 |
| all | choice | 237 | 0.810 | - | 0.691 | 0.070 | 0.879 |
| all | score | 156 | 0.263 | 0.769 | 1.358 | 0.081 | 0.248 |
| all | all | 1079 | 0.585 | - | 0.787 | 0.058 | 0.646 |

### test-tasks (401 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 24 | 0.500 | - | 0.834 | 0.263 | 0.450 |
| document | choice | 32 | 0.906 | - | 0.345 | 0.120 | 1.000 |
| document | score | 70 | 0.286 | 0.643 | 1.379 | 0.021 | 0.304 |
| photo | bool | 8 | 0.750 | - | 0.446 | 0.130 | 0.857 |
| photo | choice | 58 | 0.638 | - | 1.261 | 0.207 | 0.681 |
| photo | score | 24 | 0.292 | 0.792 | 1.379 | 0.084 | 0.350 |
| screenshot | bool | 38 | 0.632 | - | 0.602 | 0.151 | 0.677 |
| screenshot | score | 147 | 0.204 | 0.497 | 1.398 | 0.085 | 0.229 |
| document | all | 126 | 0.484 | - | 1.013 | 0.055 | 0.545 |
| photo | all | 90 | 0.556 | - | 1.220 | 0.144 | 0.625 |
| screenshot | all | 185 | 0.292 | - | 1.235 | 0.098 | 0.331 |
| all | bool | 70 | 0.600 | - | 0.664 | 0.103 | 0.625 |
| all | choice | 90 | 0.733 | - | 0.935 | 0.141 | 0.806 |
| all | score | 241 | 0.237 | 0.568 | 1.391 | 0.051 | 0.264 |
| all | all | 401 | 0.411 | - | 1.162 | 0.071 | 0.470 |

### test-styles (1202 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 451 | 0.563 | - | 0.723 | 0.105 | 0.587 |
| document | choice | 133 | 0.737 | - | 1.062 | 0.103 | 0.766 |
| document | score | 169 | 0.278 | 0.775 | 1.351 | 0.048 | 0.301 |
| screenshot | bool | 318 | 0.569 | - | 0.688 | 0.052 | 0.584 |
| screenshot | choice | 106 | 0.868 | - | 0.635 | 0.103 | 0.918 |
| screenshot | score | 25 | 0.400 | 0.680 | 1.345 | 0.126 | 0.450 |
| document | all | 753 | 0.530 | - | 0.924 | 0.084 | 0.587 |
| screenshot | all | 449 | 0.630 | - | 0.712 | 0.040 | 0.672 |
| all | bool | 769 | 0.566 | - | 0.709 | 0.082 | 0.583 |
| all | choice | 239 | 0.795 | - | 0.873 | 0.073 | 0.833 |
| all | score | 194 | 0.294 | 0.763 | 1.350 | 0.046 | 0.327 |
| all | all | 1202 | 0.567 | - | 0.845 | 0.061 | 0.626 |

### test-external (689 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 41 | 0.463 | - | 1.207 | 0.415 | 0.455 |
| document | score | 33 | 0.455 | 0.818 | 1.289 | 0.126 | 0.481 |
| photo | bool | 174 | 0.828 | - | 0.345 | 0.068 | 0.914 |
| photo | choice | 348 | 0.971 | - | 0.202 | 0.134 | 0.996 |
| screenshot | choice | 93 | 0.882 | - | 0.384 | 0.066 | 0.947 |
| document | all | 74 | 0.459 | - | 1.243 | 0.286 | 0.483 |
| photo | all | 522 | 0.923 | - | 0.250 | 0.094 | 0.969 |
| screenshot | all | 93 | 0.882 | - | 0.384 | 0.066 | 0.947 |
| all | bool | 215 | 0.758 | - | 0.509 | 0.070 | 0.814 |
| all | choice | 441 | 0.952 | - | 0.240 | 0.114 | 0.983 |
| all | score | 33 | 0.455 | 0.818 | 1.289 | 0.126 | 0.481 |
| all | all | 689 | 0.868 | - | 0.375 | 0.077 | 0.924 |
