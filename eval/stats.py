"""Small statistics helpers shared by the benchmarks (eval/tts_benchmark.py, eval/retrieval_benchmark.py)."""

from __future__ import annotations

import random
import statistics


def coefficient_of_variation(values: list[float]) -> float:
    """Sample standard deviation over the mean; 0.0 for fewer than two values."""
    if len(values) < 2:
        return 0.0
    m = statistics.fmean(values)
    return statistics.stdev(values) / m if m else 0.0


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lower, upper) with a percentile bootstrap over the values (here: sentences)."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return statistics.fmean(values), means[int(n_boot * alpha / 2)], means[int(n_boot * (1 - alpha / 2)) - 1]


def paired_ci(a: dict[str, float], b: dict[str, float], **kw) -> tuple[float, float, float, int, int, int]:
    """CI of mean(a - b) over the ids both have, plus (a better, tie, a worse) counts where lower is better."""
    ids = sorted(set(a) & set(b))
    diffs = [a[i] - b[i] for i in ids]
    mean, lo, hi = bootstrap_ci(diffs, **kw)
    better = sum(d < -1e-9 for d in diffs)
    worse = sum(d > 1e-9 for d in diffs)
    return mean, lo, hi, better, len(diffs) - better - worse, worse


def cluster_bootstrap_ci(values: list[float], clusters: list[str], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lower, upper) when the values come in correlated groups -- here the same question asked once per borrower,
    so 270 loan questions are really 30 distinct ones. Resamples whole clusters with replacement; a plain bootstrap over
    the values would treat each repeat as new evidence and report an interval several times too narrow."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    groups: dict[str, list[float]] = {}
    for v, c in zip(values, clusters):
        groups.setdefault(c, []).append(v)
    keys = sorted(groups)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        picked = [v for k in rng.choices(keys, k=len(keys)) for v in groups[k]]
        means.append(statistics.fmean(picked))
    means.sort()
    return statistics.fmean(values), means[int(n_boot * alpha / 2)], means[int(n_boot * (1 - alpha / 2)) - 1]
