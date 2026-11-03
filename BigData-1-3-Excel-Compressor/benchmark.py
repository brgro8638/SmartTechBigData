"""Benchmark helpers shared by the tests and Streamlit app."""

import statistics
import time
import tracemalloc


def time_function(function, data, repeats=3):
    """Return median runtime and result; timing stays free of tracemalloc overhead."""
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(data)
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def peak_memory(function, data):
    """Return peak allocated bytes and result from a separate tracemalloc run."""
    tracemalloc.start()
    try:
        result = function(data)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak, result


def check_requirements(encode_time, decode_time, peak_factor, decoded, original, max_seconds, max_memory_factor):
    """Evaluate K1 runtime, K2 memory, and K3 lossless roundtrip requirements."""
    return {
        "Speed K1": encode_time + decode_time <= max_seconds,
        "Memory K2": peak_factor <= max_memory_factor,
        "Lossless K3": decoded == original,
    }
