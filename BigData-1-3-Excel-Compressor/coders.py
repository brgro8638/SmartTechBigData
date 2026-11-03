"""Lossless byte-oriented coders refactored from Compression.ipynb."""

import heapq
from bisect import bisect_right
from collections import Counter

import numpy as np


def _progress_callback(progress, total):
    """Return a callback that reports at most roughly one update per input percent."""
    if progress is None or total <= 0:
        return lambda index: None
    next_report = [0]
    step = max(1, total // 100)

    def report(index):
        if index >= next_report[0]:
            progress(min(1.0, index / total))
            next_report[0] += step

    return report


def entropy(data: bytes) -> float:
    """Return entropy in bits per byte."""
    counts = Counter(data)
    n = len(data)
    probabilities = np.array(list(counts.values())) / n
    return np.sum(-probabilities * np.log2(probabilities))


def rle_simple_encode(data: bytes, progress=None) -> bytes:
    """Encode runs as [count][value] pairs, with a maximum count of 255."""
    report = _progress_callback(progress, len(data))
    output = bytearray()
    index = 0
    while index < len(data):
        count = 1
        while index + count < len(data) and data[index + count] == data[index] and count < 255:
            count += 1
        output.append(count)
        output.append(data[index])
        index += count
        report(index)
    if progress is not None:
        progress(1.0)
    return bytes(output)


def rle_simple_decode(code: bytes, progress=None) -> bytes:
    """Decode [count][value] pairs produced by rle_simple_encode."""
    report = _progress_callback(progress, len(code))
    output = bytearray()
    for index in range(0, len(code), 2):
        count, value = code[index], code[index + 1]
        output.extend(bytes([value]) * count)
        report(index + 2)
    if progress is not None:
        progress(1.0)
    return bytes(output)


def packbits_encode(data: bytes, progress=None) -> bytes:
    """Encode data using the PackBits format from the notebook."""
    report = _progress_callback(progress, len(data))
    output = bytearray()
    index = 0
    while index < len(data):
        run = 1
        while index + run < len(data) and data[index + run] == data[index] and run < 128:
            run += 1

        if run >= 3:
            output.append(256 - (run - 1))
            output.append(data[index])
            index += run
        else:
            start = index
            while index < len(data) and index - start < 128:
                if index + 2 < len(data) and data[index] == data[index + 1] == data[index + 2]:
                    break
                index += 1
            length = index - start
            output.append(length - 1)
            output.extend(data[start:index])
        report(index)
    if progress is not None:
        progress(1.0)
    return bytes(output)


def packbits_decode(code: bytes, progress=None) -> bytes:
    """Decode PackBits data."""
    report = _progress_callback(progress, len(code))
    output = bytearray()
    index = 0
    while index < len(code):
        header = code[index]
        index += 1
        if header < 128:
            output.extend(code[index:index + header + 1])
            index += header + 1
        elif header > 128:
            output.extend(bytes([code[index]]) * (257 - header))
            index += 1
        report(index)
    if progress is not None:
        progress(1.0)
    return bytes(output)


def build_huffman_tree(data):
    """Build the deterministic Huffman tree used by the notebook."""
    frequencies = Counter(data)
    heap = [(frequency, index, byte) for index, (byte, frequency) in enumerate(sorted(frequencies.items()))]
    heapq.heapify(heap)
    counter = len(heap)
    while len(heap) > 1:
        frequency_left, _, left = heapq.heappop(heap)
        frequency_right, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (frequency_left + frequency_right, counter, (left, right)))
        counter += 1
    return heap[0][2]


def make_codes(tree, prefix="", codes=None):
    """Return byte-to-bit-string codes by walking a Huffman tree."""
    if codes is None:
        codes = {}
    if isinstance(tree, int):
        codes[tree] = prefix or "0"
    else:
        left, right = tree
        make_codes(left, prefix + "0", codes)
        make_codes(right, prefix + "1", codes)
    return codes


def huffman_encode(data: bytes, progress=None) -> bytes:
    """Encode bytes with a frequency-table header and Huffman payload."""
    frequencies = Counter(data)
    codes = make_codes(build_huffman_tree(frequencies))
    report = _progress_callback(progress, len(data))
    bits = []
    for index, byte in enumerate(data, 1):
        bits.append(codes[byte])
        report(index)
    bit_string = "".join(bits)

    padding = (8 - len(bit_string) % 8) % 8
    bit_string += "0" * padding
    output = bytearray()
    output += len(frequencies).to_bytes(2, "big")
    for byte, frequency in sorted(frequencies.items()):
        output.append(byte)
        output += frequency.to_bytes(4, "big")
    output.append(padding)
    output += bytes(int(bit_string[index:index + 8], 2) for index in range(0, len(bit_string), 8))
    if progress is not None:
        progress(1.0)
    return bytes(output)


def huffman_decode(code: bytes, progress=None) -> bytes:
    """Decode a Huffman payload using its serialized frequency table."""
    symbol_count = int.from_bytes(code[0:2], "big")
    position, frequencies = 2, {}
    for _ in range(symbol_count):
        frequencies[code[position]] = int.from_bytes(code[position + 1:position + 5], "big")
        position += 5
    padding = code[position]
    position += 1
    tree = build_huffman_tree(frequencies)
    bits = "".join(f"{byte:08b}" for byte in code[position:])
    if padding:
        bits = bits[:-padding]
    report = _progress_callback(progress, len(bits))
    if isinstance(tree, int):
        output = bytes([tree]) * len(bits)
        if progress is not None:
            progress(1.0)
        return output

    output, node = bytearray(), tree
    for index, bit in enumerate(bits, 1):
        node = node[0] if bit == "0" else node[1]
        if isinstance(node, int):
            output.append(node)
            node = tree
        report(index)
    if progress is not None:
        progress(1.0)
    return bytes(output)


BITS = 32
TOP = (1 << BITS) - 1
HALF = 1 << (BITS - 1)
Q1 = 1 << (BITS - 2)
Q3 = 3 * Q1


def build_cumulative(frequencies):
    """Return sorted symbols and cumulative frequencies."""
    symbols = sorted(frequencies)
    cumulative = [0]
    for symbol in symbols:
        cumulative.append(cumulative[-1] + frequencies[symbol])
    return symbols, cumulative


def arithmetic_encode(data: bytes, progress=None) -> bytes:
    """Encode bytes with the notebook's static integer arithmetic coder."""
    frequencies = Counter(data)
    symbols, cumulative = build_cumulative(frequencies)
    total = cumulative[-1]
    index_by_symbol = {symbol: index for index, symbol in enumerate(symbols)}
    report = _progress_callback(progress, len(data))

    bits, pending = [], 0

    def emit(bit):
        nonlocal pending
        bits.append(bit)
        bits.extend([1 - bit] * pending)
        pending = 0

    low, high = 0, TOP
    for index, byte in enumerate(data, 1):
        symbol_index = index_by_symbol[byte]
        width = high - low + 1
        high = low + width * cumulative[symbol_index + 1] // total - 1
        low = low + width * cumulative[symbol_index] // total
        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF
            elif low >= Q1 and high < Q3:
                pending += 1
                low -= Q1
                high -= Q1
            else:
                break
            low = 2 * low
            high = 2 * high + 1
        report(index)

    pending += 1
    emit(0 if low < Q1 else 1)
    padding = (8 - len(bits) % 8) % 8
    bits += [0] * padding
    output = bytearray(len(data).to_bytes(4, "big"))
    output += len(symbols).to_bytes(2, "big")
    for symbol in symbols:
        output.append(symbol)
        output += frequencies[symbol].to_bytes(4, "big")
    for index in range(0, len(bits), 8):
        value = 0
        for bit in bits[index:index + 8]:
            value = (value << 1) | bit
        output.append(value)
    if progress is not None:
        progress(1.0)
    return bytes(output)


def arithmetic_decode(code: bytes, progress=None) -> bytes:
    """Decode bytes produced by arithmetic_encode."""
    length = int.from_bytes(code[0:4], "big")
    symbol_count = int.from_bytes(code[4:6], "big")
    position, frequencies = 6, {}
    for _ in range(symbol_count):
        frequencies[code[position]] = int.from_bytes(code[position + 1:position + 5], "big")
        position += 5
    symbols, cumulative = build_cumulative(frequencies)
    total = cumulative[-1]
    data_bits = code[position:]
    bit_count = len(data_bits) * 8

    def bit(index):
        return 0 if index >= bit_count else (data_bits[index >> 3] >> (7 - (index & 7))) & 1

    value = 0
    for index in range(BITS):
        value = (value << 1) | bit(index)
    bit_index, low, high, output = BITS, 0, TOP, bytearray()
    report = _progress_callback(progress, length)
    for decoded_count in range(length):
        width = high - low + 1
        target = ((value - low + 1) * total - 1) // width
        symbol_index = bisect_right(cumulative, target) - 1
        output.append(symbols[symbol_index])
        high = low + width * cumulative[symbol_index + 1] // total - 1
        low = low + width * cumulative[symbol_index] // total
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                value -= HALF
            elif low >= Q1 and high < Q3:
                low -= Q1
                high -= Q1
                value -= Q1
            else:
                break
            low = 2 * low
            high = 2 * high + 1
            value = 2 * value + bit(bit_index)
            bit_index += 1
        report(decoded_count + 1)
    if progress is not None:
        progress(1.0)
    return bytes(output)


class ContextModel:
    """Adaptive probability model for one binary decision."""

    def __init__(self):
        self.c0 = 1
        self.c1 = 1

    def update(self, bit, limit=1024):
        if bit == 0:
            self.c0 += 1
        else:
            self.c1 += 1
        if self.c0 + self.c1 > limit:
            self.c0 = (self.c0 + 1) // 2
            self.c1 = (self.c1 + 1) // 2


def context_index(mode, position, node):
    """Select position or tree context, matching the notebook."""
    return position if mode == "position" else node


def cabac_encode(data: bytes, mode="tree", progress=None) -> bytes:
    """Encode bytes with the notebook's adaptive binary arithmetic coder."""
    contexts = [ContextModel() for _ in range(256)]
    low, high, pending = 0, TOP, 0
    output_bits = []
    report = _progress_callback(progress, len(data))

    def emit(bit):
        nonlocal pending
        output_bits.append(bit)
        output_bits.extend([1 - bit] * pending)
        pending = 0

    for byte_index, byte in enumerate(data, 1):
        node = 1
        for position in range(8):
            bit = (byte >> (7 - position)) & 1
            model = contexts[context_index(mode, position, node)]
            width = high - low + 1
            split = low + width * model.c0 // (model.c0 + model.c1) - 1
            if bit == 0:
                high = split
            else:
                low = split + 1
            model.update(bit)
            node = (node << 1) | bit
            while True:
                if high < HALF:
                    emit(0)
                elif low >= HALF:
                    emit(1)
                    low -= HALF
                    high -= HALF
                elif low >= Q1 and high < Q3:
                    pending += 1
                    low -= Q1
                    high -= Q1
                else:
                    break
                low = 2 * low
                high = 2 * high + 1
        report(byte_index)

    pending += 1
    emit(0 if low < Q1 else 1)
    while len(output_bits) % 8:
        output_bits.append(0)
    payload = bytes(int("".join(map(str, output_bits[index:index + 8])), 2) for index in range(0, len(output_bits), 8))
    if progress is not None:
        progress(1.0)
    return len(data).to_bytes(4, "big") + payload


def cabac_decode(code: bytes, mode="tree", progress=None) -> bytes:
    """Decode bytes produced by cabac_encode."""
    length = int.from_bytes(code[:4], "big")
    bits = ((byte >> (7 - index)) & 1 for byte in code[4:] for index in range(8))

    def next_bit():
        return next(bits, 0)

    value = 0
    for _ in range(BITS):
        value = (value << 1) | next_bit()
    low, high = 0, TOP
    contexts = [ContextModel() for _ in range(256)]
    output = bytearray()
    report = _progress_callback(progress, length)

    for byte_index in range(length):
        node = 1
        for position in range(8):
            model = contexts[context_index(mode, position, node)]
            width = high - low + 1
            split = low + width * model.c0 // (model.c0 + model.c1) - 1
            if value <= split:
                high = split
                bit = 0
            else:
                low = split + 1
                bit = 1
            model.update(bit)
            node = (node << 1) | bit
            while True:
                if high < HALF:
                    pass
                elif low >= HALF:
                    low -= HALF
                    high -= HALF
                    value -= HALF
                elif low >= Q1 and high < Q3:
                    low -= Q1
                    high -= Q1
                    value -= Q1
                else:
                    break
                low = 2 * low
                high = 2 * high + 1
                value = 2 * value + next_bit()
        output.append(node & 0xFF)
        report(byte_index + 1)
    if progress is not None:
        progress(1.0)
    return bytes(output)


CODERS = {
    "Simple RLE": (rle_simple_encode, rle_simple_decode),
    "PackBits": (packbits_encode, packbits_decode),
    "Huffman": (huffman_encode, huffman_decode),
    "Arithmetic": (arithmetic_encode, arithmetic_decode),
    "CABAC (tree)": (cabac_encode, cabac_decode),
}
