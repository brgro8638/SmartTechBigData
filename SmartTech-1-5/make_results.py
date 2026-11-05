"""Regenerate results/benchmark_256.csv, the numbers quoted in the README and video.

The same run is available in the GUI under "Benchmark all data".
"""

from pathlib import Path

import pandas as pd

import benchmark
import prediction
import sources

OUTPUT = Path(__file__).resolve().parent / "results" / "benchmark_256.csv"
SIDE = 256

rows = benchmark.matrix_rows(
    sources.benchmark_sources(),
    lambda values: sources.limit_size(values, SIDE),
    prediction.PREDICTORS,
)
pd.DataFrame(rows).round(4).to_csv(OUTPUT, index=False)
print(f"Wrote {len(rows)} rows to {OUTPUT}")
