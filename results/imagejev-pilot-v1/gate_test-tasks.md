# Gate: imagejev:pilot_v1_best vs siglip2-zeroshot

**FAIL** (criterion 1: FAIL, criterion 2: FAIL)

## Criterion 1: lower log loss and ECE in every domain (`test-tasks`)

| domain | log loss (cand / base) | ECE (cand / base) | result |
|---|---|---|---|
| photo | 1.593 / 1.180 | 0.153 / 0.138 | FAIL |
| document | 1.002 / 1.026 | 0.090 / 0.070 | FAIL |
| screenshot | 1.284 / 1.234 | 0.219 / 0.058 | FAIL |

## Criterion 2: no domain below the baseline's accuracy on any split

| split | domain | accuracy (cand / base) | result |
|---|---|---|---|
| val | photo | 0.715 / 0.596 | PASS |
| val | document | 0.620 / 0.580 | PASS |
| val | screenshot | 0.690 / 0.614 | PASS |
| val-tasks | photo | 0.629 / 0.647 | FAIL |
| val-tasks | document | 0.547 / 0.545 | PASS |
| val-tasks | screenshot | 0.685 / 0.693 | FAIL |
| test-images | photo | 0.706 / 0.585 | PASS |
| test-images | document | 0.641 / 0.595 | PASS |
| test-images | screenshot | 0.691 / 0.629 | PASS |
| test-tasks | photo | 0.601 / 0.559 | PASS |
| test-tasks | document | 0.433 / 0.454 | FAIL |
| test-tasks | screenshot | 0.170 / 0.347 | FAIL |
| test-styles | document | 0.577 / 0.531 | PASS |
| test-styles | screenshot | 0.702 / 0.626 | PASS |
| test-external | photo | 0.936 / 0.927 | PASS |
| test-external | document | 0.469 / 0.463 | PASS |
| test-external | screenshot | 0.841 / 0.836 | PASS |
