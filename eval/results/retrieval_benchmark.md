# RAG retrieval benchmark

Read the header of `eval/retrieval_benchmark.py` for what is measured, the decision rule and the limits. Intervals are 95% percentile bootstrap. 'Deployed view' is Recall@2, because `check_policy` hands the agent two chunks.

## Policy questions: dev set (settings were chosen here) (n=38)

| setting | Recall@1 | Recall@2 (deployed view) | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| pool 4 (as deployed) | 0.63 [0.47, 0.79] | 0.76 [0.63, 0.89] | 0.84 [0.71, 0.95] | 0.84 [0.71, 0.95] | 0.72 [0.60, 0.84] |
| pool 4, zero-overlap keyword matches allowed (old behaviour) | 0.63 [0.47, 0.79] | 0.76 [0.63, 0.89] | 0.84 [0.71, 0.95] | 0.84 [0.71, 0.95] | 0.72 [0.60, 0.84] |
| pool 8 | 0.63 [0.47, 0.79] | 0.79 [0.66, 0.92] | 0.87 [0.74, 0.97] | 0.89 [0.79, 0.97] | 0.74 [0.62, 0.85] |
| pool 16 | 0.63 [0.47, 0.79] | 0.79 [0.66, 0.92] | 0.87 [0.74, 0.97] | 0.89 [0.79, 0.97] | 0.74 [0.62, 0.85] |
| pool 4 + headings in search text | 0.66 [0.50, 0.82] | 0.76 [0.63, 0.89] | 0.82 [0.68, 0.92] | 0.84 [0.71, 0.95] | 0.73 [0.60, 0.85] |
| pool 4 + headings everywhere | 0.66 [0.50, 0.82] | 0.76 [0.63, 0.89] | 0.82 [0.68, 0.92] | 0.87 [0.76, 0.97] | 0.74 [0.61, 0.86] |
| pool 4 + rank fusion | 0.63 [0.47, 0.76] | 0.76 [0.63, 0.89] | 0.84 [0.71, 0.95] | 0.87 [0.76, 0.97] | 0.73 [0.60, 0.84] |
| pool 4, no query translation | 0.50 [0.34, 0.66] | 0.68 [0.53, 0.84] | 0.76 [0.63, 0.89] | 0.76 [0.63, 0.89] | 0.62 [0.48, 0.75] |

## Policy questions: held-out set (confirms the choice) (n=47)

| setting | Recall@1 | Recall@2 (deployed view) | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| pool 4 (as deployed) | 0.85 [0.74, 0.94] | 0.94 [0.85, 1.00] | 0.96 [0.89, 1.00] | 0.98 [0.94, 1.00] | 0.91 [0.83, 0.97] |
| pool 4, zero-overlap keyword matches allowed (old behaviour) | 0.85 [0.74, 0.94] | 0.94 [0.85, 1.00] | 0.96 [0.89, 1.00] | 0.98 [0.94, 1.00] | 0.91 [0.83, 0.97] |
| pool 8 | 0.85 [0.74, 0.94] | 0.94 [0.85, 1.00] | 0.96 [0.89, 1.00] | 0.98 [0.94, 1.00] | 0.91 [0.83, 0.97] |
| pool 16 | 0.85 [0.74, 0.94] | 0.94 [0.85, 1.00] | 0.96 [0.89, 1.00] | 0.96 [0.89, 1.00] | 0.90 [0.82, 0.97] |
| pool 4 + headings in search text | 0.81 [0.68, 0.91] | 0.94 [0.87, 1.00] | 0.96 [0.89, 1.00] | 0.98 [0.94, 1.00] | 0.88 [0.81, 0.95] |
| pool 4 + headings everywhere | 0.81 [0.68, 0.91] | 0.94 [0.87, 1.00] | 0.96 [0.89, 1.00] | 0.98 [0.94, 1.00] | 0.88 [0.81, 0.95] |
| pool 4 + rank fusion | 0.85 [0.74, 0.94] | 0.94 [0.85, 1.00] | 0.98 [0.94, 1.00] | 0.98 [0.94, 1.00] | 0.91 [0.84, 0.97] |
| pool 4, no query translation | 0.77 [0.64, 0.87] | 0.85 [0.74, 0.94] | 0.87 [0.77, 0.96] | 0.91 [0.83, 0.98] | 0.83 [0.73, 0.92] |

## Loan-agreement facts, own agreement or the right policy chunk (n=270, 30 distinct questions asked once per borrower; intervals resample those)

| setting | Recall@1 | Recall@2 (deployed view) | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| pool 4 (as deployed) | 0.83 [0.70, 0.94] | 0.93 [0.84, 0.99] | 0.93 [0.84, 0.99] | 0.93 [0.86, 0.99] | 0.88 [0.78, 0.96] |
| pool 4, zero-overlap keyword matches allowed (old behaviour) | 0.83 [0.70, 0.94] | 0.93 [0.84, 0.99] | 0.93 [0.84, 0.99] | 0.93 [0.86, 0.99] | 0.88 [0.78, 0.96] |
| pool 8 | 0.85 [0.72, 0.96] | 0.97 [0.90, 1.00] | 0.97 [0.90, 1.00] | 0.97 [0.90, 1.00] | 0.91 [0.82, 0.98] |
| pool 16 | 0.85 [0.72, 0.96] | 0.97 [0.90, 1.00] | 0.97 [0.90, 1.00] | 0.97 [0.90, 1.00] | 0.91 [0.82, 0.98] |
| pool 4 + headings in search text | 0.83 [0.69, 0.94] | 0.93 [0.84, 0.99] | 0.93 [0.86, 0.99] | 0.93 [0.86, 0.99] | 0.88 [0.78, 0.96] |
| pool 4 + headings everywhere | 0.84 [0.70, 0.96] | 0.95 [0.87, 1.00] | 0.95 [0.88, 1.00] | 0.96 [0.91, 1.00] | 0.90 [0.81, 0.98] |
| pool 4 + rank fusion | 0.84 [0.72, 0.94] | 0.92 [0.84, 0.99] | 0.93 [0.84, 0.99] | 0.93 [0.86, 0.99] | 0.89 [0.79, 0.96] |
| pool 4, no query translation | 0.66 [0.50, 0.81] | 0.77 [0.63, 0.90] | 0.79 [0.64, 0.90] | 0.83 [0.69, 0.94] | 0.73 [0.59, 0.85] |

## Each change against its parent: paired difference [95% interval] (questions won-lost)

| setting | parent | Recall@2 dev | Recall@2 held-out | Recall@2 loan | MRR dev | MRR held-out | MRR loan | verdict |
|---|---|---|---|---|---|---|---|---|
| pool 4, zero-overlap keyword matches allowed (old behaviour) | pool 4 (as deployed) | +0.00 [+0.00, +0.00] (0-0) | +0.00 [+0.00, +0.00] (0-0) | +0.00 [+0.00, +0.00] (0-0) | +0.00 [+0.00, +0.00] (0-0) | +0.00 [+0.00, +0.00] (0-0) | +0.00 [+0.00, +0.00] (0-0) | no measurable difference -- not shipped |
| pool 8 | pool 4 (as deployed) | +0.03 [+0.00, +0.08] (1-0) | +0.00 [+0.00, +0.00] (0-0) | +0.04 [+0.00, +0.09] (10-0) | +0.02 [+0.00, +0.05] (2-0) | -0.00 [-0.00, +0.00] (0-1) | +0.03 [-0.00, +0.07] (10-1) | no measurable difference -- not shipped |
| pool 16 | pool 8 | +0.00 [+0.00, +0.00] (0-0) | +0.00 [+0.00, +0.00] (0-0) | +0.00 [+0.00, +0.00] (0-0) | +0.00 [-0.02, +0.02] (1-1) | -0.00 [-0.01, +0.00] (0-1) | +0.00 [+0.00, +0.00] (0-0) | no measurable difference -- not shipped |
| pool 4 + headings in search text | pool 4 (as deployed) | +0.00 [+0.00, +0.00] (0-0) | +0.00 [-0.06, +0.06] (1-1) | +0.00 [+0.00, +0.00] (0-0) | +0.01 [-0.01, +0.04] (1-1) | -0.02 [-0.06, +0.00] (1-3) | +0.00 [-0.03, +0.03] (7-6) | no measurable difference -- not shipped |
| pool 4 + headings everywhere | pool 4 (as deployed) | +0.00 [-0.08, +0.08] (1-1) | +0.00 [-0.06, +0.06] (1-1) | +0.02 [-0.01, +0.06] (6-1) | +0.02 [-0.06, +0.09] (3-2) | -0.02 [-0.06, +0.00] (1-3) | +0.02 [-0.03, +0.08] (16-7) | no measurable difference -- not shipped |
| pool 4 + rank fusion | pool 4 (as deployed) | +0.00 [-0.13, +0.13] (3-3) | +0.00 [-0.06, +0.06] (1-1) | -0.01 [-0.02, +0.00] (0-2) | +0.01 [-0.08, +0.09] (5-4) | +0.00 [-0.04, +0.05] (3-2) | +0.01 [-0.04, +0.06] (19-16) | no measurable difference -- not shipped |
| pool 4, no query translation | pool 4 (as deployed) | -0.08 [-0.18, +0.03] (1-4) | -0.09 [-0.17, -0.02] (0-4) | -0.16 [-0.29, -0.03] (0-42) | -0.11 [-0.21, -0.02] (2-7) | -0.08 [-0.16, -0.02] (0-5) | -0.15 [-0.27, -0.05] (0-54) | hurts -- not shipped |

## Isolation and latency

| setting | results from another borrower's documents | retrieve() p50 / p95 ms |
|---|---|---|
| pool 4 (as deployed) | 0 | 255.9 / 329.3 (n=40) |
| pool 4, zero-overlap keyword matches allowed (old behaviour) | 0 | 255.6 / 307.2 (n=40) |
| pool 8 | 0 | 378.0 / 447.5 (n=40) |
| pool 16 | 0 | 570.8 / 608.7 (n=40) |
| pool 4 + headings in search text | 0 | 271.8 / 362.9 (n=40) |
| pool 4 + headings everywhere | 0 | 257.6 / 323.4 (n=40) |
| pool 4 + rank fusion | 0 | 252.4 / 325.7 (n=40) |
| pool 4, no query translation | 0 | 244.5 / 315.8 (n=40) |

Isolation: 0 expected; 0 across 8 settings x 270 questions.

## Where the loan questions still miss (pool 4 (as deployed), Recall@2)

| fact | principal | tenure | emi | rate | late_fee | settlement | first_emi | security | court | extension |
|---|---|---|---|---|---|---|---|---|---|---|
| Recall@2 | 1.00 | 0.67 | 0.85 | 1.00 | 1.00 | 1.00 | 1.00 | 0.93 | 0.85 | 1.00 |

By phrasing: clean 1.00, colloquial 0.79, hinglish 1.00.

## Can the scores tell an answerable question from an unanswerable one?

| setting | AUC of the top reranker score | answerable n / unanswerable n |
|---|---|---|
| pool 4 (as deployed) | 0.8855 | 85 / 30 |
| pool 4, zero-overlap keyword matches allowed (old behaviour) | 0.8855 | 85 / 30 |
| pool 8 | 0.8863 | 85 / 30 |
| pool 16 | 0.8863 | 85 / 30 |
| pool 4 + headings in search text | 0.8882 | 85 / 30 |
| pool 4 + headings everywhere | 0.8875 | 85 / 30 |
| pool 4 + rank fusion | 0.8675 | 85 / 30 |
| pool 4, no query translation | 0.7898 | 79 / 28 |

### The 'nothing relevant' threshold (pool 4 (as deployed))

Rule: the highest top-result score that keeps 95% of the dev questions = **-9.35** (a cross-encoder logit). Checked on questions it was not set from:

| should be kept | kept | | should be dropped | dropped |
|---|---|---|---|---|
| dev (n=38) | 97% | | all unanswerable (n=30) | 47% |
| held-out (n=47) | 98% | | off-topic | 70% |
| loan facts (n=270) | 97% | | near-topic (in the lending domain, absent from the documents) | 35% |


Fixtures (sha256): dev `13b30b5a0bb6`, heldout `f2276edb85ab`, negative `6f742fa3feb2`, loan_accounts `10cfabb5da4f`
