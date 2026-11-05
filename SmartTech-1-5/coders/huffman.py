"""Canonical static Huffman coding.

The Huffman tree only fixes each symbol's code *length*. Canonical codes are then
assigned by (length, symbol), so the header needs nothing but the lengths.
"""

from __future__ import annotations

import heapq
from collections import Counter

import numpy as np

from .common import (
    Encoded,
    check_image,
    read_shape,
    read_varint,
    unzigzag,
    write_shape,
    write_varint,
    zigzag,
)


def code_lengths(frequencies: dict[int, int]) -> dict[int, int]:
    """Merge the two rarest groups repeatedly; each merge adds one bit to its members."""
    if len(frequencies) == 1:
        # One symbol has no tree, but it still needs one bit to be counted.
        return {next(iter(frequencies)): 1}

    # The index breaks ties, which keeps the result deterministic.
    heap = [(count, index, [symbol]) for index, (symbol, count) in enumerate(sorted(frequencies.items()))]
    heapq.heapify(heap)
    lengths = dict.fromkeys(frequencies, 0)
    next_index = len(heap)
    while len(heap) > 1:
        count_a, _, group_a = heapq.heappop(heap)
        count_b, _, group_b = heapq.heappop(heap)
        for symbol in group_a + group_b:
            lengths[symbol] += 1
        heapq.heappush(heap, (count_a + count_b, next_index, group_a + group_b))
        next_index += 1
    return lengths


def canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    """Return symbol -> (code value, code length)."""
    codes = {}
    code = 0
    previous_length = 0
    for symbol, length in sorted(lengths.items(), key=lambda item: (item[1], item[0])):
        code <<= length - previous_length
        codes[symbol] = (code, length)
        code += 1
        previous_length = length
    return codes


def code_table(symbols: np.ndarray) -> dict[int, str]:
    """Readable symbol -> bit string table, used by the GUI."""
    frequencies = Counter(symbols.ravel().tolist())
    if not frequencies:
        return {}
    codes = canonical_codes(code_lengths(frequencies))
    return {symbol: format(code, f"0{length}b") for symbol, (code, length) in codes.items()}


def encode(image: np.ndarray) -> Encoded:
    image = check_image(image)
    symbols = image.ravel().tolist()
    header = bytearray()
    write_shape(header, image.shape)
    if not symbols:
        return Encoded(bytes(header), len(header), {"alphabet": 0})

    frequencies = Counter(symbols)
    lengths = code_lengths(frequencies)
    codes = canonical_codes(lengths)

    # Header: alphabet size, then sorted symbols as deltas with their code lengths.
    ordered = sorted(lengths)
    write_varint(header, len(ordered))
    write_varint(header, zigzag(ordered[0]))
    previous = ordered[0]
    for symbol in ordered:
        write_varint(header, symbol - previous)
        write_varint(header, lengths[symbol])
        previous = symbol

    strings = {symbol: format(code, f"0{length}b") for symbol, (code, length) in codes.items()}
    bit_string = "".join(map(strings.__getitem__, symbols))
    padding = -len(bit_string) % 8
    header.append(padding)
    bit_string += "0" * padding
    payload = int(bit_string, 2).to_bytes(len(bit_string) // 8, "big")

    max_length = max(lengths.values())
    stats = {
        "alphabet": len(ordered),
        "max_code_length": max_length,
        "mean_code_length": sum(frequencies[s] * lengths[s] for s in ordered) / len(symbols),
    }
    return Encoded(bytes(header) + payload, len(header), stats)


def decode(stream: bytes) -> np.ndarray:
    shape, position = read_shape(stream, 0)
    count = shape[0] * shape[1]
    if count == 0:
        return np.zeros(shape, dtype=np.int64)

    alphabet, position = read_varint(stream, position)
    first, position = read_varint(stream, position)
    symbol = unzigzag(first)
    lengths = {}
    for _ in range(alphabet):
        delta, position = read_varint(stream, position)
        symbol += delta
        length, position = read_varint(stream, position)
        lengths[symbol] = length
    position += 1  # The padding byte is implied by the known symbol count.

    # Keyed on (length, code): codes are prefix-free, so the first hit is the symbol.
    lookup = {(length, code): symbol for symbol, (code, length) in canonical_codes(lengths).items()}

    output = [0] * count
    filled = 0
    code = 0
    length = 0
    for byte in stream[position:]:
        for shift in range(7, -1, -1):
            code = (code << 1) | ((byte >> shift) & 1)
            length += 1
            symbol = lookup.get((length, code))
            if symbol is not None:
                output[filled] = symbol
                filled += 1
                code = 0
                length = 0
                if filled == count:
                    return np.array(output, dtype=np.int64).reshape(shape)
    raise ValueError("Huffman stream ended before all symbols were decoded")
