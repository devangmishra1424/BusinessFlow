# Guardrail benchmark: invented figures and links

Read the header of `eval/guardrail_benchmark.py` for how the cases are built and what they cannot show.

- Fabricated amounts and links caught: **417/417 (100.0%, 95% CI 99.1%-100.0%)**
- Grounded amounts and links wrongly blocked: **0/258 (0.0%, 95% CI 0.0%-1.5%)**
- Correct figures the model derived itself (blocked by design; the check compares values, it does not do arithmetic): 5/5 (100.0%, 95% CI 56.6%-100.0%)

Fabricated caught, by how the amount was written:

| format | caught |
|---|---|
| Devanagari digits | 50/50 |
| Hindi suffix | 50/50 |
| INR | 50/50 |
| Indian grouping | 50/50 |
| Rs no dot | 50/50 |
| Rs. | 50/50 |
| invented link | 7/7 |
| lakh | 10/10 |
| rupee sign | 50/50 |
| rupees after | 50/50 |

Grounded passed, by how the amount was written:

| format | passed |
|---|---|
| Devanagari digits | 30/30 |
| Hindi suffix | 30/30 |
| INR | 30/30 |
| Indian grouping | 30/30 |
| Rs no dot | 30/30 |
| Rs. | 30/30 |
| lakh | 5/5 |
| no figure | 3/3 |
| real link | 6/6 |
| rounded | 4/4 |
| rupee sign | 30/30 |
| rupees after | 30/30 |
