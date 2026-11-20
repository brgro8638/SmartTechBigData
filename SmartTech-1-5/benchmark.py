"""Run the coders and measure what the GUI reports."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import prediction
from coders import CODERS, cabac


def entropy_bits(symbols: np.ndarray) -> float:
    """Order-0 Shannon entropy per symbol: the limit for a static order-0 coder."""
    if symbols.size == 0:
        return 0.0
    _, counts = np.unique(symbols, return_counts=True)
    probabilities = counts / symbols.size
    return float(-(probabilities * np.log2(probabilities)).sum())


@dataclass
class CoderResult:
    coder: str
    total_bytes: int
    header_bytes: int
    payload_bytes: int
    bits_per_pixel: float
    ratio: float
    saving_percent: float
    gap_to_entropy_percent: float
    encode_seconds: float
    decode_seconds: float
    lossless: bool
    stats: dict = field(default_factory=dict)
    stream_head: bytes = b""


def cabac_mode(predictor: str) -> int:
    """Residuals are residuals even if a smooth image happens to give no negative one."""
    return cabac.MODE_RAW if predictor == "none" else cabac.MODE_RESIDUAL


def run_coder(name: str, pixels: np.ndarray, predictor: str, nominal_bits: int) -> CoderResult:
    """Encode, decode from the bytes alone, invert prediction and compare to the input."""
    module = CODERS[name]
    symbols = prediction.forward(pixels, predictor)

    started = time.perf_counter()
    encoded = cabac.encode(symbols, cabac_mode(predictor)) if name == "CABAC" else module.encode(symbols)
    encode_seconds = time.perf_counter() - started

    started = time.perf_counter()
    decoded = module.decode(encoded.stream)
    decode_seconds = time.perf_counter() - started

    restored = prediction.inverse(decoded, predictor)
    lossless = restored.shape == pixels.shape and bool(np.array_equal(restored, pixels))

    size = max(pixels.size, 1)
    total = len(encoded.stream)
    raw_bytes = pixels.size * nominal_bits / 8
    entropy = entropy_bits(symbols)
    bpp = total * 8 / size
    return CoderResult(
        coder=name,
        total_bytes=total,
        header_bytes=encoded.header_bytes,
        payload_bytes=encoded.payload_bytes,
        bits_per_pixel=bpp,
        ratio=raw_bytes / total if total else 0.0,
        saving_percent=(1 - total / raw_bytes) * 100 if raw_bytes else 0.0,
        gap_to_entropy_percent=(bpp - entropy) / entropy * 100 if entropy else float("nan"),
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
        lossless=lossless,
        stats=encoded.stats,
        stream_head=encoded.stream[encoded.header_bytes:encoded.header_bytes + 32],
    )


def run_all(pixels: np.ndarray, predictor: str, nominal_bits: int) -> list[CoderResult]:
    return [run_coder(name, pixels, predictor, nominal_bits) for name in CODERS]


def matrix_rows(sources_list: list, size_limiter, predictors: dict, on_step=None) -> list[dict]:
    """Run every source x predictor x coder. Returns one flat row per run."""
    rows = []
    total = len(sources_list) * len(predictors) * len(CODERS)
    done = 0
    for source in sources_list:
        pixels = size_limiter(source.pixels)
        for label, predictor in predictors.items():
            entropy = entropy_bits(prediction.forward(pixels, predictor))
            for name in CODERS:
                result = run_coder(name, pixels, predictor, source.nominal_bits)
                rows.append(
                    {
                        "Category": source.category,
                        "Source": source.name,
                        "Size": f"{pixels.shape[1]}x{pixels.shape[0]}",
                        "Nominal bits": source.nominal_bits,
                        "Prediction": label,
                        "Entropy H0 (bits)": entropy,
                        "Coder": name,
                        "Bits/pixel": result.bits_per_pixel,
                        "Ratio": result.ratio,
                        "Total bytes": result.total_bytes,
                        "Header bytes": result.header_bytes,
                        "Encode s": result.encode_seconds,
                        "Decode s": result.decode_seconds,
                        "Lossless": result.lossless,
                    }
                )
                done += 1
                if on_step:
                    on_step(done / total, f"{source.name}: {name}, {label}")
    return rows
