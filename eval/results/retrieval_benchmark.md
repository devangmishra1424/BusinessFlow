# RAG retrieval benchmark

Read the header of `eval/retrieval_benchmark.py` for what is measured, the decision rule and the limits. Intervals are 95% percentile bootstrap. 'Deployed view' is Recall@2, because `check_policy` hands the agent two chunks.

## Policy questions: dev set (settings were chosen here) (n=38)

| setting | Recall@1 | Recall@2 (deployed view) | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| pool 4 (as deployed) | 0.61 [0.45, 0.76] | 0.71 [0.55, 0.84] | 0.79 [0.63, 0.92] | 0.82 [0.68, 0.92] | 0.69 [0.55, 0.82] |
| pool 4, zero-overlap keyword matches allowed (old behaviour) | 0.61 [0.45, 0.76] | 0.71 [0.55, 0.84] | 0.79 [0.63, 0.92] | 0.82 [0.68, 0.92] | 0.69 [0.55, 0.82] |
| pool 8 | 0.66 [0.50, 0.82] | 0.79 [0.66, 0.92] | 0.82 [0.68, 0.92] | 0.89 [0.79, 0.97] | 0.75 [0.63, 0.86] |
| pool 16 | 0.66 [0.50, 0.82] | 0.79 [0.66, 0.92] | 0.82 [0.68, 0.92] | 0.89 [0.79, 0.97] | 0.75 [0.63, 0.86] |
| pool 4 + headings in search text | 0.61 [0.45, 0.76] | 0.71 [0.55, 0.84] | 0.79 [0.63, 0.92] | 0.82 [0.68, 0.92] | 0.69 [0.55, 0.82] |
| pool 4 + headings everywhere | 0.66 [0.50, 0.82] | 0.79 [0.66, 0.92] | 0.84 [0.71, 0.95] | 0.87 [0.76, 0.97] | 0.75 [0.62, 0.86] |
| pool 4 + rank fusion | 0.61 [0.45, 0.76] | 0.76 [0.61, 0.89] | 0.79 [0.63, 0.92] | 0.82 [0.68, 0.92] | 0.70 [0.56, 0.82] |
| pool 4, no query translation | 0.47 [0.32, 0.63] | 0.63 [0.47, 0.79] | 0.68 [0.53, 0.82] | 0.76 [0.61, 0.89] | 0.59 [0.44, 0.72] |

## Policy questions: held-out set (confirms the choice) (n=47)

| setting | Recall@1 | Recall@2 (deployed view) | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| pool 4 (as deployed) | 0.83 [0.72, 0.94] | 0.94 [0.85, 1.00] | 0.94 [0.85, 1.00] | 0.96 [0.89, 1.00] | 0.89 [0.81, 0.96] |
| pool 4, zero-overlap keyword matches allowed (old behaviour) | 0.83 [0.72, 0.94] | 0.94 [0.85, 1.00] | 0.94 [0.85, 1.00] | 0.96 [0.89, 1.00] | 0.89 [0.81, 0.96] |
| pool 8 | 0.83 [0.72, 0.94] | 0.94 [0.85, 1.00] | 0.96 [0.89, 1.00] | 0.98 [0.94, 1.00] | 0.90 [0.82, 0.96] |
| pool 16 | 0.83 [0.72, 0.94] | 0.94 [0.85, 1.00] | 0.96 [0.89, 1.00] | 0.96 [0.89, 1.00] | 0.89 [0.81, 0.96] |
| pool 4 + headings in search text | 0.79 [0.66, 0.89] | 0.89 [0.81, 0.98] | 0.91 [0.83, 0.98] | 0.94 [0.85, 1.00] | 0.85 [0.76, 0.93] |
| pool 4 + headings everywhere | 0.79 [0.66, 0.89] | 0.91 [0.83, 0.98] | 0.94 [0.85, 1.00] | 0.96 [0.89, 1.00] | 0.86 [0.78, 0.94] |
| pool 4 + rank fusion | 0.79 [0.66, 0.89] | 0.96 [0.89, 1.00] | 0.96 [0.89, 1.00] | 0.96 [0.89, 1.00] | 0.87 [0.79, 0.94] |
| pool 4, no query translation | 0.74 [0.62, 0.87] | 0.85 [0.74, 0.94] | 0.87 [0.77, 0.96] | 0.94 [0.85, 1.00] | 0.82 [0.72, 0.91] |

## Loan-agreement facts, own agreement or the right policy chunk (n=270, 30 distinct questions asked once per borrower; intervals resample those)

| setting | Recall@1 | Recall@2 (deployed view) | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| pool 4 (as deployed) | 0.81 [0.69, 0.93] | 0.92 [0.82, 0.99] | 0.92 [0.83, 0.99] | 0.92 [0.83, 0.99] | 0.87 [0.77, 0.95] |
| pool 4, zero-overlap keyword matches allowed (old behaviour) | 0.81 [0.69, 0.93] | 0.92 [0.82, 0.99] | 0.92 [0.83, 0.99] | 0.92 [0.83, 0.99] | 0.87 [0.77, 0.95] |
| pool 8 | 0.84 [0.71, 0.95] | 0.97 [0.90, 1.00] | 0.97 [0.90, 1.00] | 0.97 [0.90, 1.00] | 0.90 [0.81, 0.97] |
| pool 16 | 0.84 [0.71, 0.95] | 0.97 [0.90, 1.00] | 0.97 [0.90, 1.00] | 0.97 [0.90, 1.00] | 0.90 [0.81, 0.97] |
| pool 4 + headings in search text | 0.82 [0.68, 0.94] | 0.92 [0.83, 0.99] | 0.92 [0.83, 0.99] | 0.92 [0.83, 0.99] | 0.87 [0.77, 0.95] |
| pool 4 + headings everywhere | 0.84 [0.70, 0.96] | 0.94 [0.86, 1.00] | 0.96 [0.90, 1.00] | 0.96 [0.90, 1.00] | 0.90 [0.81, 0.97] |
| pool 4 + rank fusion | 0.84 [0.71, 0.94] | 0.91 [0.82, 0.99] | 0.92 [0.83, 0.99] | 0.92 [0.83, 0.99] | 0.88 [0.78, 0.96] |
| pool 4, no query translation | 0.66 [0.51, 0.80] | 0.76 [0.61, 0.90] | 0.79 [0.64, 0.91] | 0.85 [0.73, 0.95] | 0.73 [0.60, 0.85] |

## Each change against its parent: paired difference [95% interval] (questions won-lost)

| setting | parent | Recall@2 dev | Recall@2 held-out | Recall@2 loan | MRR dev | MRR held-out | MRR loan | verdict |
|---|---|---|---|---|---|---|---|---|
| pool 4, zero-overlap keyword matches allowed (old behaviour) | pool 4 (as deployed) | +0.00 [+0.00, +0.00] (0-0) | +0.00 [+0.00, +0.00] (0-0) | +0.00 [+0.00, +0.00] (0-0) | +0.00 [+0.00, +0.00] (0-0) | +0.00 [+0.00, +0.00] (0-0) | +0.00 [+0.00, +0.00] (0-0) | no measurable difference -- not shipped |
| pool 8 | pool 4 (as deployed) | +0.08 [+0.00, +0.18] (3-0) | +0.00 [+0.00, +0.00] (0-0) | +0.05 [+0.00, +0.12] (13-0) | +0.06 [-0.01, +0.15] (4-3) | +0.01 [+0.00, +0.02] (1-0) | +0.03 [-0.00, +0.08] (13-1) | no measurable difference -- not shipped |
| pool 16 | pool 8 | +0.00 [+0.00, +0.00] (0-0) | +0.00 [+0.00, +0.00] (0-0) | +0.00 [+0.00, +0.00] (0-0) | +0.00 [-0.02, +0.02] (1-1) | -0.01 [-0.02, +0.00] (0-1) | +0.00 [+0.00, +0.00] (0-0) | no measurable difference -- not shipped |
| pool 4 + headings in search text | pool 4 (as deployed) | +0.00 [+0.00, +0.00] (0-0) | -0.04 [-0.11, +0.00] (0-2) | +0.00 [+0.00, +0.01] (1-0) | +0.00 [+0.00, +0.00] (0-0) | -0.04 [-0.09, +0.00] (0-3) | +0.00 [-0.03, +0.03] (9-7) | no measurable difference -- not shipped |
| pool 4 + headings everywhere | pool 4 (as deployed) | +0.08 [+0.00, +0.18] (3-0) | -0.02 [-0.09, +0.04] (1-2) | +0.02 [-0.01, +0.08] (7-1) | +0.06 [+0.00, +0.14] (4-0) | -0.02 [-0.08, +0.02] (1-3) | +0.03 [-0.03, +0.10] (18-8) | helps on a set, but the held-out set does not confirm it -- not shipped |
| pool 4 + rank fusion | pool 4 (as deployed) | +0.05 [-0.05, +0.16] (3-1) | +0.02 [+0.00, +0.06] (1-0) | -0.00 [-0.01, +0.00] (0-1) | +0.01 [-0.07, +0.09] (5-3) | -0.02 [-0.06, +0.03] (2-3) | +0.01 [-0.04, +0.06] (20-15) | no measurable difference -- not shipped |
| pool 4, no query translation | pool 4 (as deployed) | -0.08 [-0.18, +0.03] (1-4) | -0.09 [-0.17, -0.02] (0-4) | -0.16 [-0.29, -0.03] (0-42) | -0.10 [-0.21, -0.01] (2-7) | -0.07 [-0.14, -0.01] (0-4) | -0.13 [-0.24, -0.05] (0-51) | hurts -- not shipped |

## Isolation and latency

| setting | results from another borrower's documents | retrieve() p50 / p95 ms |
|---|---|---|
| pool 4 (as deployed) | 0 | 197.1 / 256.7 (n=40) |
| pool 4, zero-overlap keyword matches allowed (old behaviour) | 0 | 192.7 / 251.1 (n=40) |
| pool 8 | 0 | 367.9 / 436.8 (n=40) |
| pool 16 | 0 | 537.7 / 642.1 (n=40) |
| pool 4 + headings in search text | 0 | 214.3 / 253.7 (n=40) |
| pool 4 + headings everywhere | 0 | 204.4 / 253.3 (n=40) |
| pool 4 + rank fusion | 0 | 199.4 / 252.8 (n=40) |
| pool 4, no query translation | 0 | 191.4 / 255.4 (n=40) |

Isolation: 0 expected; 0 across 8 settings x 270 questions.

## Where the loan questions still miss (pool 4 (as deployed), Recall@2)

| fact | principal | tenure | emi | rate | late_fee | settlement | first_emi | security | court | extension |
|---|---|---|---|---|---|---|---|---|---|---|
| Recall@2 | 1.00 | 0.67 | 0.85 | 1.00 | 1.00 | 1.00 | 1.00 | 0.93 | 0.74 | 1.00 |

By phrasing: clean 1.00, colloquial 0.76, hinglish 1.00.

## Can the scores tell an answerable question from an unanswerable one?

| setting | AUC of the top reranker score | answerable n / unanswerable n |
|---|---|---|
| pool 4 (as deployed) | 0.8816 | 85 / 30 |
| pool 4, zero-overlap keyword matches allowed (old behaviour) | 0.8816 | 85 / 30 |
| pool 8 | 0.8855 | 85 / 30 |
| pool 16 | 0.8855 | 85 / 30 |
| pool 4 + headings in search text | 0.8851 | 85 / 30 |
| pool 4 + headings everywhere | 0.8851 | 85 / 30 |
| pool 4 + rank fusion | 0.8596 | 85 / 30 |
| pool 4, no query translation | 0.7857 | 79 / 28 |

### The 'nothing relevant' threshold (pool 4 (as deployed))

Rule: the highest top-result score that keeps 95% of the dev questions = **-9.52** (a cross-encoder logit). Checked on questions it was not set from:

| should be kept | kept | | should be dropped | dropped |
|---|---|---|---|---|
| dev (n=38) | 97% | | all unanswerable (n=30) | 47% |
| held-out (n=47) | 98% | | off-topic | 70% |
| loan facts (n=270) | 97% | | near-topic (in the lending domain, absent from the documents) | 35% |


Fixtures (sha256): dev `13b30b5a0bb6`, heldout `f5560934ddb1`, negative `da852757c0c1`, loan_accounts `0b07325cb63e`
