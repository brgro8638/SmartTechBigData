"""Static order-0 arithmetic coding with a 32-bit integer interval.

The message narrows an interval [low, high]. Whenever the leading bit of low and
high agree it can never change, so it is sent and the interval is doubled. That
is why a symbol can cost a fraction of a bit, which Huffman cannot do.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter

import numpy as np

from .common import (
    BitReader,
    Encoded,
    bits_to_bytes,
    check_image,
    read_shape,
    read_varint,
    unzigzag,
    write_shape,
    write_varint,
    zigzag,
)

PRECISION = 32
TOP = (1 << PRECISION) - 1
HALF = 1 << (PRECISION - 1)
QUARTER = 1 << (PRECISION - 2)
THREE_QUARTERS = 3 * QUARTER
# Above this total, a narrow interval could give a symbol zero width.
MAX_TOTAL = QUARTER


def _cumulative(frequencies: dict[int, int]) -> tuple[list[int], list[int]]:
    ordered = sorted(frequencies)
    cumulative = [0]
    for symbol in ordered:
        cumulative.append(cumulative[-1] + frequencies[symbol])
    return ordered, cumulative


def encode(image: np.ndarray) -> Encoded:
    image = check_image(image)
    symbols = image.ravel().tolist()
    if len(symbols) >= MAX_TOTAL:
        raise ValueError("Input is too large for 32-bit arithmetic coding precision")

    header = bytearray()
    write_shape(header, image.shape)
    if not symbols:
        return Encoded(bytes(header), len(header), {"alphabet": 0})

    frequencies = Counter(symbols)
    ordered, cumulative = _cumulative(frequencies)

    # The decoder must rebuild the exact same model, so the counts are sent.
    write_varint(header, len(ordered))
    write_varint(header, zigzag(ordered[0]))
    previous = ordered[0]
    for symbol in ordered:
        write_varint(header, symbol - previous)
        write_varint(header, frequencies[symbol])
        previous = symbol

    total = cumulative[-1]
    index_of = {symbol: index for index, symbol in enumerate(ordered)}
    low, high, pending = 0, TOP, 0
    bits: list[int] = []

    for symbol in symbols:
        index = index_of[symbol]
        width = high - low + 1
        high = low + width * cumulative[index + 1] // total - 1
        low = low + width * cumulative[index] // total
        while True:
            if high < HALF:
                bits.append(0)
                if pending:
                    bits.extend([1] * pending)
                    pending = 0
            elif low >= HALF:
                bits.append(1)
                if pending:
                    bits.extend([0] * pending)
                    pending = 0
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                # Straddling the middle: the next bit is unknown, so remember it as pending.
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1

    # Two more bits pin down a value inside the final interval.
    pending += 1
    final_bit = 0 if low < QUARTER else 1
    bits.append(final_bit)
    bits.extend([1 - final_bit] * pending)

    stats = {"alphabet": len(ordered), "bits_emitted": len(bits)}
    return Encoded(bytes(header) + bits_to_bytes(bits), len(header), stats)


def decode(stream: bytes) -> np.ndarray:
    shape, position = read_shape(stream, 0)
    count = shape[0] * shape[1]
    if count == 0:
        return np.zeros(shape, dtype=np.int64)

    alphabet, position = read_varint(stream, position)
    first, position = read_varint(stream, position)
    symbol = unzigzag(first)
    frequencies = {}
    for _ in range(alphabet):
        delta, position = read_varint(stream, position)
        symbol += delta
        frequencies[symbol], position = read_varint(stream, position)

    ordered, cumulative = _cumulative(frequencies)
    total = cumulative[-1]
    reader = BitReader(stream, position)
    read = reader.read

    value = 0
    for _ in range(PRECISION):
        value = (value << 1) | read()
    low, high = 0, TOP
    output = [0] * count

    for filled in range(count):
        width = high - low + 1
        target = ((value - low + 1) * total - 1) // width
        index = bisect_right(cumulative, target) - 1
        output[filled] = ordered[index]
        high = low + width * cumulative[index + 1] // total - 1
        low = low + width * cumulative[index] // total
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                value -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                low -= QUARTER
                high -= QUARTER
                value -= QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            value = (value << 1) | read()

    return np.array(output, dtype=np.int64).reshape(shape)


def trace(image: np.ndarray, limit: int) -> list[dict]:
    """Replay the first `limit` symbols with the real model and log the interval.

    The model is the one built from the whole image, as in encode().
    """
    all_symbols = image.ravel().tolist()
    if not all_symbols:
        return []
    frequencies = Counter(all_symbols)
    ordered, cumulative = _cumulative(frequencies)
    total = cumulative[-1]
    index_of = {symbol: index for index, symbol in enumerate(ordered)}

    low, high, pending, emitted = 0, TOP, 0, 0
    ideal = 0.0
    rows = []
    for step, symbol in enumerate(all_symbols[:limit], 1):
        index = index_of[symbol]
        width = high - low + 1
        high = low + width * cumulative[index + 1] // total - 1
        low = low + width * cumulative[index] // total
        probability = frequencies[symbol] / total
        ideal += -math.log2(probability)
        low_before, high_before = low, high
        new_bits = ""
        while True:
            if high < HALF:
                new_bits += "0" + "1" * pending
                pending = 0
            elif low >= HALF:
                new_bits += "1" + "0" * pending
                pending = 0
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
        emitted += len(new_bits)
        rows.append(
            {
                "step": step,
                "symbol": symbol,
                "p": probability,
                "low": f"{low_before:#010x}",
                "high": f"{high_before:#010x}",
                "new bits": new_bits or "–",
                "pending": pending,
                "bits sent": emitted,
                "ideal bits": ideal,
            }
        )
    return rows
