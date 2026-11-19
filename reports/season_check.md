# Season-level check: does parameter uncertainty help?

For 5 past seasons, simulated from after ~10 and ~20 games per team and scored the
title / top-4 / relegation probabilities against the real final tables (lower is better).
`scale` 0 = point estimate (no uncertainty), 1 = Laplace-approximation draws, >1 = wider.
Only 10 season/cut-off cases, so differences are small and noisy.

## Overall
| scale | brier | log_loss |
|---|---|---|
| 0.0000 | 0.0445 | 0.1428 |
| 0.5000 | 0.0449 | 0.1425 |
| 1.0000 | 0.0447 | 0.1409 |
| 1.5000 | 0.0449 | 0.1430 |

## By event
| scale | event | brier | log_loss |
|---|---|---|---|
| 0.0000 | relegated | 0.0580 | 0.1807 |
| 0.0000 | title | 0.0124 | 0.0422 |
| 0.0000 | top4 | 0.0631 | 0.2055 |
| 0.5000 | relegated | 0.0592 | 0.1822 |
| 0.5000 | title | 0.0118 | 0.0415 |
| 0.5000 | top4 | 0.0636 | 0.2038 |
| 1.0000 | relegated | 0.0594 | 0.1824 |
| 1.0000 | title | 0.0123 | 0.0440 |
| 1.0000 | top4 | 0.0623 | 0.1964 |
| 1.5000 | relegated | 0.0600 | 0.1862 |
| 1.5000 | title | 0.0136 | 0.0488 |
| 1.5000 | top4 | 0.0612 | 0.1940 |
