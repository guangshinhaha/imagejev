# smolvlm:SmolVLM-500M-Instruct

Evaluated on a fixed subset of at most 300 images per split.

Calibration fitted on `val-tasks`: choice=1.78, score=20.00, bool=4.62, bool_bias=0.32

- cold latency: p50 122.2 ms, p95 124.0 ms (n=30)
- warm latency: p50 44.2 ms, p95 47.6 ms (n=30)

### val-tasks (406 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 106 | 0.566 | - | 0.669 | 0.053 | 0.600 |
| document | choice | 119 | 0.479 | - | 1.152 | 0.095 | 0.542 |
| document | score | 18 | 0.333 | 0.444 | 1.363 | 0.061 | 0.333 |
| photo | bool | 48 | 0.646 | - | 0.616 | 0.144 | 0.641 |
| photo | score | 16 | 0.188 | 0.625 | 1.387 | 0.079 | 0.231 |
| screenshot | bool | 79 | 0.608 | - | 0.639 | 0.126 | 0.625 |
| screenshot | score | 20 | 0.200 | 0.600 | 1.405 | 0.076 | 0.188 |
| document | all | 243 | 0.506 | - | 0.957 | 0.069 | 0.549 |
| photo | all | 64 | 0.531 | - | 0.809 | 0.128 | 0.596 |
| screenshot | all | 99 | 0.525 | - | 0.793 | 0.116 | 0.613 |
| all | bool | 233 | 0.597 | - | 0.648 | 0.066 | 0.604 |
| all | choice | 119 | 0.479 | - | 1.152 | 0.095 | 0.542 |
| all | score | 54 | 0.241 | 0.556 | 1.386 | 0.031 | 0.227 |
| all | all | 406 | 0.515 | - | 0.894 | 0.058 | 0.575 |

### test-images (1489 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 277 | 0.560 | - | 0.692 | 0.067 | 0.563 |
| document | choice | 93 | 0.806 | - | 0.672 | 0.221 | 0.880 |
| document | score | 212 | 0.189 | 0.542 | 1.399 | 0.088 | 0.188 |
| photo | bool | 104 | 0.856 | - | 0.446 | 0.140 | 0.893 |
| photo | choice | 46 | 0.609 | - | 1.012 | 0.211 | 0.595 |
| photo | score | 150 | 0.167 | 0.413 | 1.402 | 0.114 | 0.158 |
| screenshot | bool | 296 | 0.561 | - | 0.710 | 0.096 | 0.553 |
| screenshot | choice | 208 | 0.423 | - | 1.351 | 0.219 | 0.437 |
| screenshot | score | 103 | 0.194 | 0.544 | 1.394 | 0.101 | 0.193 |
| document | all | 582 | 0.464 | - | 0.946 | 0.071 | 0.528 |
| photo | all | 300 | 0.473 | - | 1.011 | 0.102 | 0.525 |
| screenshot | all | 607 | 0.451 | - | 1.046 | 0.116 | 0.510 |
| all | bool | 677 | 0.606 | - | 0.662 | 0.048 | 0.616 |
| all | choice | 347 | 0.550 | - | 1.124 | 0.083 | 0.576 |
| all | score | 465 | 0.183 | 0.501 | 1.399 | 0.097 | 0.169 |
| all | all | 1489 | 0.461 | - | 1.000 | 0.068 | 0.532 |

### test-tasks (395 questions, 300 images)
| domain | type | n | acc | within-1 | log loss | ECE | acc@80% |
|---|---|---|---|---|---|---|---|
| document | bool | 22 | 0.455 | - | 0.766 | 0.354 | 0.333 |
| document | choice | 32 | 0.406 | - | 1.606 | 0.128 | 0.462 |
| document | score | 70 | 0.314 | 0.600 | 1.386 | 0.041 | 0.321 |
| photo | bool | 7 | 0.571 | - | 0.687 | 0.140 | 0.500 |
| photo | choice | 58 | 0.379 | - | 1.239 | 0.191 | 0.426 |
| photo | score | 24 | 0.208 | 0.417 | 1.396 | 0.078 | 0.250 |
| screenshot | bool | 35 | 0.743 | - | 0.560 | 0.132 | 0.821 |
| screenshot | score | 147 | 0.184 | 0.456 | 1.409 | 0.102 | 0.203 |
| document | all | 124 | 0.363 | - | 1.333 | 0.097 | 0.360 |
| photo | all | 89 | 0.348 | - | 1.238 | 0.147 | 0.403 |
| screenshot | all | 182 | 0.291 | - | 1.246 | 0.108 | 0.336 |
| all | bool | 64 | 0.625 | - | 0.645 | 0.183 | 0.615 |
| all | choice | 90 | 0.389 | - | 1.369 | 0.125 | 0.444 |
| all | score | 241 | 0.224 | 0.494 | 1.401 | 0.056 | 0.223 |
| all | all | 395 | 0.327 | - | 1.271 | 0.062 | 0.358 |
